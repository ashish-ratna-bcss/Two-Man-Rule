# main.py
import sys
import multiprocessing as mp
mp.current_process().authkey = b'pmj_auth'
from datetime import datetime, timezone, timedelta
IST = timezone(timedelta(hours=5, minutes=30))
import os
import cv2
import numpy as np
import argparse
import time
import math
import threading
import signal
import subprocess
import re
# Heavy torch-backed imports (PoseDetector, PersonTracker) are deferred into
# main() so the process holds ZERO GPU VRAM until its window starts. Keeping
# them at module top would trigger torch/ultralytics/supervision import on
# every spawn, which initializes the CUDA driver context and reserves
# ~50–500 MiB even while idle.
from models.door_verifier import DoorVerifier
from logic.roi_manager import ROIManager
from logic.state_machine import DualAuthStateMachine

from io_.video_handler import VideoHandler
from io_.frame_quality import FrameQualityGate
from io_.visualizer import Visualizer
from io_.alert_system import AlertSystem
from io_.runtime_logger import RuntimeEventLogger
from io_.terminal_tee import enable_terminal_capture
import config
import json
from io_.frame_timing_tracker import FrameTimingTracker, FrameTimingEvent

# Global reference for cleanup in single-stream mode
detector = None



def _start_stream_process(cmd):
    kwargs = {}
    if os.name != "nt":
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def _terminate_stream_process(proc, timeout=3.0):
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        pass
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            pass
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass


def _stop_gpu_process(proc, timeout=3.0):
    if proc is None:
        return
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
    if proc.is_alive() and hasattr(proc, "kill"):
        proc.kill()
        proc.join(timeout=1)



def _parse_stream_indices(indices_text: str, total_streams: int) -> list:
    parsed = []
    seen   = set()
    for token in indices_text.split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit():
            raise ValueError(f"Invalid stream index '{token}'.")
        idx = int(token)
        if idx < 0 or idx >= total_streams:
            raise ValueError(f"Invalid stream-index {idx}. Available: 0 to {total_streams - 1}.")
        if idx not in seen:
            parsed.append(idx)
            seen.add(idx)
    if not parsed:
        raise ValueError("No valid stream indexes provided.")
    return parsed


def _parse_stream_video_overrides(override_specs: list, total_streams: int) -> dict:
    overrides = {}
    for spec in override_specs:
        idx_text, separator, video_source = spec.partition("=")
        idx_text = idx_text.strip()
        video_source = video_source.strip()
        if not separator or not idx_text or not video_source:
            raise ValueError(
                f"Invalid --stream-video '{spec}'. Use INDEX=VIDEO_PATH."
            )
        if not idx_text.isdigit():
            raise ValueError(f"Invalid --stream-video stream index '{idx_text}'.")

        idx = int(idx_text)
        if idx < 0 or idx >= total_streams:
            raise ValueError(f"Invalid stream-index {idx}. Available: 0 to {total_streams - 1}.")
        if idx in overrides:
            raise ValueError(f"Duplicate --stream-video override for stream {idx}.")
        overrides[idx] = video_source
    return overrides


def _infer_test_window_from_video_source(video_source) -> str:
    if not isinstance(video_source, str):
        return None
    filename = os.path.splitext(os.path.basename(video_source))[0]
    suffix = filename.rsplit("-", 1)[-1].upper()
    return {"M": "morning", "E": "evening"}.get(suffix)


def _video_source_is_live(video_source) -> bool:
    if video_source is None or isinstance(video_source, int):
        return True
    return isinstance(video_source, str) and (
        video_source.isdigit() or video_source.lower().startswith("rtsp://")
    )


class _NumpySafeEncoder(json.JSONEncoder):
    """Handles numpy scalar types that are not natively JSON serializable."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


CALIBRATED_W, CALIBRATED_H = 2688, 1520

_alert_counter = [0]  # mutable so capture() can increment across calls


def _install_term_handler():
    """Child process SIGTERM/SIGINT → flush stdio, exit 0.

    Supervisor sends SIGTERM via process group when shutting down. Without a
    handler the default action kills the process mid-loop, leaving partial
    log lines and an undefined VideoHandler / detector. Handler raises
    SystemExit so the `with VideoHandler(...)` __exit__ + finally blocks run.
    """
    def _handler(signum, _frame):
        print(f"[SYSTEM] Received signal {signum}. Exiting cleanly.")
        sys.exit(0)
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _handler)
        except Exception:
            pass


def _scale_polygon(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale RTSP-calibrated polygon coordinates to another source frame size."""
    scale = np.array([width / CALIBRATED_W, height / CALIBRATED_H], dtype=np.float32)
    return np.rint(points.astype(np.float32) * scale).astype(np.int32)


def setup_rois(roi_manager: ROIManager, stream_rois: dict, width: int, height: int, scale_rois: bool = False):
    """Register ROIs in the same coordinate space as the original video frame."""
    transform = (lambda points: _scale_polygon(points, width, height)) if scale_rois else (
        lambda points: points.astype(np.int32).copy()
    )
    rois = {
        "LOCKS_ROI":        transform(stream_rois["LOCKS_ROI"]),
        "DOOR_ROI":         transform(stream_rois["DOOR_ROI"]),
        "STANDING_ZONE":    transform(stream_rois["STANDING_ZONE"]),
        "INTERACTION_ZONE": transform(stream_rois["INTERACTION_ZONE"]),
        "DOOR_CORNER_ROI":  transform(stream_rois["DOOR_CORNER_ROI"]),
    }
    for name, points in rois.items():
        roi_manager.register_polygon_roi(name, points)
    return rois


def print_roi_coordinates(rois: dict, width: int, height: int, scale_rois: bool):
    mode = "scaled" if scale_rois else "raw RTSP-captured"
    print(f"[ROI] Using {mode} polygon coordinates:")
    for name, points in rois.items():
        coords = [tuple(map(int, pt)) for pt in points.reshape(-1, 2)]
        print(f"  {name}: {coords}")

    if not scale_rois:
        max_x = max(int(points[:, 0].max()) for points in rois.values())
        max_y = max(int(points[:, 1].max()) for points in rois.values())
        if max_x >= width or max_y >= height:
            print(
                "[WARNING] Some ROI coordinates are outside this video frame. "
                "Use --scale-rois only for downscaled test clips."
            )


def draw_rois(visualizer: Visualizer, frame: np.ndarray, rois: dict):
    roi_styles = {
        "INTERACTION_ZONE": ((100, 100, 255), 1),
        "STANDING_ZONE":    ((0, 200, 255), 1),
        "LOCKS_ROI":        ((0, 255, 255), 2),
        "DOOR_ROI":         ((255, 0, 0), 1),
        "DOOR_CORNER_ROI":  ((255, 255, 255), 2),
    }
    for name, points in rois.items():
        color, thickness = roi_styles[name]
        visualizer.draw_roi_polygon(frame, points, color, thickness)
        visualizer.draw_roi_label(frame, name, points, color)


def _bbox_height(bbox) -> float:
    if bbox is None or len(bbox) < 4:
        return 0.0
    return float(bbox[3] - bbox[1])


def _bbox_size_matches(ref_bbox, candidate_bbox, tolerance: float = 0.4) -> bool:
    ref_h = _bbox_height(ref_bbox)
    if ref_h <= 0:
        return True
    cand_h = _bbox_height(candidate_bbox)
    if cand_h <= 0:
        return False
    ratio = abs(cand_h - ref_h) / ref_h
    return ratio <= tolerance


def _label_verified_slot(
    slot: str,
    slot_label: str,
    state_machine: DualAuthStateMachine,
    tracked_persons: dict,
    labels: dict,
    frame: np.ndarray = None,
):
    session = state_machine.session
    primary_id = session.get(f"id_{slot}")
    if primary_id is None:
        return

    primary_id = int(primary_id)
    tag = f"P{1 if slot == 'a' else 2}_unlocker"
    other_tag = f"P{2 if slot == 'a' else 1}_unlocker"

    def _is_other_unlocker(tid):
        return state_machine.unlocker_tags.get(tid) == other_tag

    if primary_id in tracked_persons:
        labels[primary_id] = f"{slot_label} ID {primary_id}"
        return

    for alt_id in state_machine.get_all_ids_for_tag(tag):
        if alt_id in tracked_persons and alt_id not in labels:
            labels[alt_id] = f"{slot_label} (alt ID {alt_id})"
            print(f"[VIZ] {slot_label} alt-ID {alt_id} (primary={primary_id})")
            return

    candidates = {
        tid: p for tid, p in tracked_persons.items()
        if tid not in labels and not _is_other_unlocker(tid)
    }
    if not candidates:
        return

    anchor = state_machine.verified_anchors.get(slot)
    ref_bbox = state_machine.last_seen_bbox.get(slot)
    _h = getattr(state_machine, "slot_height_ref", {}).get(slot)
    height_ref_bbox = ref_bbox if (_h is None or (hasattr(_h, '__len__') and len(_h) == 0)) else _h
    if anchor is None:
        return

    if ref_bbox is not None:
        search_origin = ((ref_bbox[0] + ref_bbox[2]) / 2, float(ref_bbox[3]))
    else:
        search_origin = anchor

    frames_lost = state_machine.slot_lost_frames.get(slot, 0)
    wide_threshold = min(config.UNLOCKER_ANCHOR_MATCH_PIXELS * 5 + frames_lost * 12, 700)

    best_dist = float("inf")
    best_tid = None
    best_bbox = None

    for tid, person in candidates.items():
        bbox = person.get("bbox")
        if bbox is None:
            continue
        cx = (bbox[0] + bbox[2]) / 2
        cy = float(bbox[3])
        dist = math.dist((cx, cy), search_origin)
        if dist < best_dist:
            best_dist = dist
            best_tid = tid
            best_bbox = bbox

    if best_tid is not None and best_dist <= wide_threshold:
        if not _bbox_size_matches(height_ref_bbox, best_bbox, tolerance=0.4):
            return
        labels[best_tid] = f"{slot_label} (remapped ID {primary_id})"
        state_machine.assign_unlocker_tag(best_tid, slot)
        print(f"[VIZ] {slot_label} live-remapped → ID {best_tid} (dist={best_dist:.1f}, lost={frames_lost}f)")


def get_unlocker_labels(
    state_machine: DualAuthStateMachine,
    tracked_persons: dict = None,
    frame: np.ndarray = None,
) -> dict:
    session = state_machine.session
    labels = {}
    tracked_persons = tracked_persons or {}

    if session.get("candidate_a") is not None:
        labels[int(session["candidate_a"])] = f"P1 unlocking ID {int(session['candidate_a'])}"
    if session.get("candidate_b") is not None:
        labels[int(session["candidate_b"])] = f"P2 unlocking ID {int(session['candidate_b'])}"

    _label_verified_slot("a", "P1 verified", state_machine, tracked_persons, labels, frame)
    _label_verified_slot("b", "P2 verified", state_machine, tracked_persons, labels, frame)

    return labels


def draw_lost_verified_ghosts(
    visualizer: Visualizer,
    frame: np.ndarray,
    state_machine: DualAuthStateMachine,
    unlocker_labels: dict,
):
    for slot in ("a", "b"):
        if state_machine.session.get(f"id_{slot}") is None:
            anchor = state_machine.verified_anchors.get(slot)
            if anchor is not None:
                label = f"RECOVERING P{1 if slot == 'a' else 2}..."
                visualizer.draw_ghost_anchor(frame, anchor, label)


def draw_pose_debug(frame: np.ndarray, tracked_persons: dict, visible_ids: set):
    keypoint_styles = {
        0:                           ("HEAD", (255, 255, 0)),
        config.KEYPOINT_WRIST_LEFT:  ("LW",   (0, 255, 0)),
        config.KEYPOINT_WRIST_RIGHT: ("RW",   (0, 255, 0)),
        config.KEYPOINT_ELBOW_LEFT:  ("LE",   (0, 165, 255)),
        config.KEYPOINT_ELBOW_RIGHT: ("RE",   (0, 165, 255)),
    }
    for track_id, person in tracked_persons.items():
        if track_id not in visible_ids:
            continue
        keypoints = person.get("keypoints")
        if keypoints is None:
            continue
        for idx, (label, color) in keypoint_styles.items():
            if idx >= len(keypoints):
                continue
            x, y, conf = keypoints[idx]
            if conf < 0.25:
                continue
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)
            cv2.putText(
                frame, label, (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
            )


def can_show_live_window(show_live: bool) -> bool:
    if not show_live:
        return False
    if os.name == "posix" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        print("[WARNING] No display server detected; live window disabled.")
        return False
    return True


def should_freeze_for_frame_quality(frame_quality_result, frame_quality_active: bool) -> bool:
    return bool(
        frame_quality_active
        and frame_quality_result is not None
        and not frame_quality_result.usable
    )


def _seconds_until_next_window(skip_active: bool = False) -> float:
    """Return seconds until the next morning (07:00) or evening (19:00) window begins.

    Returns 0.0 immediately if we are already inside an active window, UNLESS
    skip_active=True — then the currently-active window is treated as already
    consumed and the result counts to the *following* window. This lets a
    process that has already completed today's audit for the active window go
    back to sleep instead of busy-restarting inside the same window.
    """
    now = datetime.now(IST)
    curr_min = now.hour * 60 + now.minute

    morning_start_min = 7 * 60    # 07:00
    morning_end_min   = 12 * 60   # 12:00 (extended from 11:00: late openings ~11:15 were missed)
    evening_start_min = 19 * 60   # 19:00
    evening_end_min   = 23 * 60   # 23:00

    # Already inside an active window → no sleep needed (unless skipping it)
    if morning_start_min <= curr_min <= morning_end_min:
        if not skip_active:
            return 0.0
        # Morning consumed → next is evening 19:00 today
        target = now.replace(hour=19, minute=0, second=0, microsecond=0)
    elif evening_start_min <= curr_min <= evening_end_min:
        if not skip_active:
            return 0.0
        # Evening consumed → next is morning 07:00 tomorrow
        tomorrow = now + timedelta(days=1)
        target = tomorrow.replace(hour=7, minute=0, second=0, microsecond=0)
    # Before morning window → sleep until 07:00 today
    elif curr_min < morning_start_min:
        target = now.replace(hour=7, minute=0, second=0, microsecond=0)
    # Between windows (12:01 – 18:59) → sleep until 19:00 today
    elif curr_min < evening_start_min:
        target = now.replace(hour=19, minute=0, second=0, microsecond=0)
    # After evening window (23:01+) → sleep until 07:00 tomorrow
    else:
        tomorrow = now + timedelta(days=1)
        target = tomorrow.replace(hour=7, minute=0, second=0, microsecond=0)

    return max(0.0, (target - now).total_seconds())


def _current_window_name(now: datetime = None) -> str:
    """Return 'morning'/'evening' if now is inside an audit window, else None."""
    now = now or datetime.now(IST)
    curr_min = now.hour * 60 + now.minute
    if 7 * 60 <= curr_min <= 11 * 60:
        return "morning"
    if 19 * 60 <= curr_min <= 23 * 60:
        return "evening"
    return None


# ── Durable per-stream, per-(date,window) completion markers ──────────────────
# Once a stream finishes its morning/evening audit, it writes a marker file. A
# respawn (supervisor / systemd / cron) that lands inside the same window reads
# the marker and sleeps to the NEXT window instead of re-running the audit and
# emitting duplicate captures. Markers are date+window keyed, so a new day
# naturally clears them. In-memory flags alone cannot survive sys.exit(0).
def _completion_marker_path(camera_id: str, date_str: str, window: str) -> str:
    return os.path.join(
        config.BASE_LOG_DIR, "window_state", camera_id, f"{date_str}_{window}.done"
    )


def _window_already_complete(camera_id: str, date_str: str, window: str) -> bool:
    if not window:
        return False
    return os.path.exists(_completion_marker_path(camera_id, date_str, window))


def _mark_window_complete(camera_id: str, date_str: str, window: str) -> None:
    if not window:
        return
    path = _completion_marker_path(camera_id, date_str, window)
    try:
        marker_dir = os.path.dirname(path)
        os.makedirs(marker_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write(datetime.now(IST).isoformat())
        # Prune stale markers from previous days to keep the dir small.
        for name in os.listdir(marker_dir):
            if not name.startswith(date_str) and name.endswith(".done"):
                try:
                    os.remove(os.path.join(marker_dir, name))
                except OSError:
                    pass
        print(f"[SYSTEM] {camera_id}: {window} audit marked complete for {date_str}.")
    except Exception as e:
        print(f"[SYSTEM] {camera_id}: failed to write completion marker: {e}")



def capture(
    alert_system: AlertSystem,
    clean_frame: np.ndarray,
    event_type: str,
    evidence_dir: str,
    cam_id: str,
    site_name: str,
    site_id: str = "",
    details: dict = None,
    check_type: str = "System",
    visualizer: "Visualizer" = None,
    unlocker_labels: dict = None,
    tracked_persons: dict = None,
    auth_result: dict = None,
    is_door_open: bool = False,
    persons_auth_status=None,
    runtime_logger: RuntimeEventLogger = None,
    frame_idx: int = None,
    frame_ist: datetime = None,
):
    """Unified capture + log helper. Saves annotated frame + paired JSON metadata."""
    # Pin timestamp to when the frame was read from the video, not when processing finished.
    # Inference + FSM lag can be 20ms–8s; using frame_ist keeps the alert time accurate.
    now_ist  = frame_ist if frame_ist is not None else datetime.now(IST)
    date_str = now_ist.strftime("%d-%m-%Y")
    time_str = now_ist.strftime("%H-%M-%S")

    target_dir = os.path.join(evidence_dir, date_str)
    os.makedirs(target_dir, exist_ok=True)

    _alert_counter[0] += 1
    filename  = f"alert_{site_id}_{cam_id}_{date_str}_{time_str}.png"
    full_path = os.path.join(target_dir, filename)

    client_frame = clean_frame.copy()
    if visualizer is not None:
        visualizer.draw_client_overlays(
            client_frame,
            unlocker_labels or {},
            tracked_persons or {},
            auth_result or {"authorized": False},
            is_door_open,
            persons_auth_status=persons_auth_status,
        )

    ok = cv2.imwrite(full_path, client_frame)

    json_path  = full_path.rsplit('.', 1)[0] + '.json'
    event_data = {
        "site_id":  site_id,
        "cam_id":   cam_id,
        "window":   check_type.lower(),
        "events": [
            {
                "timestamp":  now_ist.isoformat(),
                "event_type": event_type,
                "details":    details or {},
            }
        ],
    }
    try:
        with open(json_path, "w") as f:
            json.dump(event_data, f, indent=4, cls=_NumpySafeEncoder)
        json_ok = True
    except Exception as e:
        print(f"[ERROR] Failed to save JSON metadata: {e}")
        json_ok = False

    if runtime_logger is not None:
        runtime_logger.write_event(
            event_type="CAPTURE",
            message=f"Capture saved for {event_type}",
            level="INFO" if ok and json_ok else "ERROR",
            details={
                "source_event_type": event_type,
                "check_type":        check_type,
                "site_name":         site_name,
                "site_id":           site_id,
                "cam_id":            cam_id,
                "image_path":        full_path,
                "json_path":         json_path,
                "image_ok":          bool(ok),
                "json_ok":           bool(json_ok),
                "event_details":     details or {},
            },
            frame_idx=frame_idx,
            ts_ist=now_ist,
        )
    print(
        f"[CAPTURE] {event_type}: {full_path} "
        f"(image={'OK' if ok else 'FAILED'}, json={'OK' if json_ok else 'FAILED'})"
    )
    return full_path


def main(
    stream_config: dict,
    video_source: str = None,
    show_live: bool = True,
    scale_rois: bool = False,
    process_every: int = 2,
    device: str = "auto",
    half: bool = True,
    show_all_detections: bool = False,
    test_window: str = None,
    debug: bool = False,
):
    global detector
    _install_term_handler()
    video_source = video_source or stream_config["rtsp_url"]
    cam_id       = stream_config["camera_id"]
    site_name    = stream_config["site_name"]

    # Strict window-gated startup. Done FIRST, before any RTSP open / log file /
    # torch import. Zero file handles, zero RTSP socket, zero VRAM until the
    # window is imminent. test_window / debug / show_all_detections bypass.
    if not test_window and not debug and not show_all_detections:
        # If a respawn lands inside a window this stream already completed today,
        # treat that window as consumed and sleep to the next one. Done BEFORE the
        # torch import below, so a completed window holds zero VRAM until the next.
        _today = datetime.now(IST).strftime("%Y-%m-%d")
        _active_win = _current_window_name()
        if _active_win and _window_already_complete(cam_id, _today, _active_win):
            secs_to_window = _seconds_until_next_window(skip_active=True)
            print(
                f"[SYSTEM] {cam_id}: {_active_win} audit already completed for {_today}. "
                f"Sleeping {secs_to_window:.0f}s to next window. Zero VRAM held."
            )
            time.sleep(secs_to_window)
        else:
            secs_to_window = _seconds_until_next_window()
            if secs_to_window > 30.0:
                print(
                    f"[SYSTEM] {cam_id}: outside auth window. Sleeping {secs_to_window:.0f}s "
                    f"before opening RTSP + loading torch. Zero VRAM held."
                )
                time.sleep(secs_to_window)

    evidence_dir = os.path.join(config.BASE_OUTPUT_DIR, site_name, cam_id)

    # Window is imminent / open. Now import torch-backed modules. This is the
    # first place CUDA driver may initialize for this process.
    from models.pose_detector import PoseDetector
    from models.tracker import PersonTracker
    from models.reid_extractor import ReIDExtractor

    runtime_logger = RuntimeEventLogger(
        base_dir=config.BASE_LOG_DIR,
        site_name=site_name,
        camera_id=cam_id,
    )

    print(f"[SYSTEM] Initializing Two-Man Rule Monitoring System for {site_name} - {cam_id}...")
    _alert_counter[0] = 0
    os.makedirs(evidence_dir, exist_ok=True)
    # Redact RTSP password before persisting to JSONL.
    _safe_video_source = re.sub(r"://[^@/]+:[^@/]+@", "://***:***@", str(video_source))
    runtime_logger.write_event(
        event_type="STREAM_START",
        message="Stream worker initialized",
        level="INFO",
        details={
            "video_source": _safe_video_source,
            "evidence_dir": evidence_dir,
            "log_file":     runtime_logger.current_file_path,
        },
    )

    # Eager detector load — eliminates the ~5-6s dead zone at window start caused
    # by lazy loading (torch import + YOLO load + 3x warmup) happening inside the
    # frame loop on first inference activation.
    _detector_load_start = time.perf_counter()
    print(f"[{cam_id}] Pre-loading detector before window opens...")
    detector = PoseDetector(device=device, half=half)
    _detector_load_ms = (time.perf_counter() - _detector_load_start) * 1000.0
    _window_ready_ist = datetime.now(IST).strftime("%H:%M:%S")
    _secs_remaining = _seconds_until_next_window()
    print(
        f"[{cam_id}] Detector ready at {_window_ready_ist} IST "
        f"(load: {_detector_load_ms:.0f}ms). "
        f"Window opens in {_secs_remaining:.0f}s."
    )
    runtime_logger.write_event(
        event_type="DETECTOR_READY",
        message="Detector pre-loaded before window activation",
        level="INFO",
        details={
            "load_ms":         round(_detector_load_ms),
            "ready_at_ist":    _window_ready_ist,
            "secs_to_window":  round(_secs_remaining),
            "device":          device,
            "half":            half,
        },
    )

    tracker      = PersonTracker()
    # Deep appearance Re-ID extractor — same per-stream/per-window lifecycle as the
    # detector (freed by the window-end sys.exit CUDA teardown). Degrades to the
    # tracker's pose-keypoint Re-ID if the weights file is absent.
    reid_extractor = ReIDExtractor(device=device)
    roi_manager  = ROIManager()

    # LKG (last-known-good) detection cache — guards against inference timeouts.
    # When detector.detect() returns None (timeout sentinel), we reuse this
    # cache rather than treating the frame as "no persons present".
    lkg_detections: list = []
    lkg_consecutive_timeouts: int = 0
    LKG_MAX_AGE = getattr(config, "LKG_MAX_CONSECUTIVE_TIMEOUTS", 15)

    with VideoHandler(video_source) as video:
        fps         = video.get_fps()
        width, height = video.get_dimensions()
        total_frames = video.get_total_frames()
        process_every = max(int(process_every), 1)
        print(f"[VIDEO] FPS: {fps:.1f}, Resolution: {width}x{height}, Total Frames: {total_frames}")
        print("[VIDEO] Processing original frames without resizing.")
        print(f"[VIDEO] Pose inference every {process_every} frame(s).")

        active_rois = setup_rois(roi_manager, stream_config["rois"], width, height, scale_rois=scale_rois)
        print_roi_coordinates(active_rois, width, height, scale_rois)

        # Presence-detection ROI: bounding box of INTERACTION_ZONE (fallback: full frame)
        _iz = active_rois.get("INTERACTION_ZONE")
        if _iz is not None and len(_iz) >= 2:
            _px1 = max(0, int(_iz[:, 0].min()))
            _py1 = max(0, int(_iz[:, 1].min()))
            _px2 = min(width,  max(_px1 + 1, int(_iz[:, 0].max())))
            _py2 = min(height, max(_py1 + 1, int(_iz[:, 1].max())))
        else:
            _px1, _py1, _px2, _py2 = 0, 0, width, height
        presence_roi_bbox = (_px1, _py1, _px2, _py2)

        # Background subtractor for cheap morning presence detection (CPU, <1ms/frame)
        presence_bg = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False
        )
        PRESENCE_WARMUP_FRAMES   = max(1, int(getattr(config, "PRESENCE_WARMUP_SECONDS", 4.0) * fps))
        PRESENCE_PIXEL_THRESHOLD = int(getattr(config, "PRESENCE_PIXEL_THRESHOLD", 2500))

        try:
            stream_ssim_thresh      = float(stream_config.get("ssim_threshold", config.SSIM_THRESHOLD))
            stream_intensity_thresh = stream_config.get("intensity_threshold", None)
            stream_motion_thresh    = stream_config.get("motion_threshold", None)
            stream_debounce         = int(stream_config.get("debounce_threshold", config.DOOR_DEBOUNCE_FRAMES))
            stream_min_visible_ratio = float(
                stream_config.get("door_corner_min_visible_ratio", config.DOOR_CORNER_MIN_VISIBLE_RATIO)
            )

            stream_ssim_thresh       = min(max(stream_ssim_thresh, 0.05), 0.99)
            if stream_intensity_thresh is not None:
                stream_intensity_thresh = float(max(stream_intensity_thresh, 0.0))
            if stream_motion_thresh is not None:
                stream_motion_thresh = float(max(stream_motion_thresh, 0.0))
            stream_debounce          = int(min(max(stream_debounce, 1), 600))
            stream_min_visible_ratio = min(max(stream_min_visible_ratio, 0.0), 1.0)

            stream_darkening = stream_config.get("darkening_protection", config.DOOR_DARKENING_PROTECTION)
            stream_open_hysteresis = float(
                stream_config.get("door_open_hysteresis", config.DOOR_OPEN_HYSTERESIS)
            )
            stream_open_hysteresis = min(max(stream_open_hysteresis, 0.0), 0.5)

            door_verifier = DoorVerifier(
                stream_config["closed_door_reference"],
                door_corner_roi=active_rois["DOOR_CORNER_ROI"],
                similarity_threshold=stream_ssim_thresh,
                debounce_threshold=stream_debounce,
                intensity_threshold=stream_intensity_thresh,
                motion_threshold=stream_motion_thresh,
                darkening_protection=bool(stream_darkening),
                min_visible_ratio=stream_min_visible_ratio,
                open_hysteresis=stream_open_hysteresis,
            )
            print(
                f"[SYSTEM] Door verifier loaded with threshold {stream_ssim_thresh} "
                f"and min_visible_ratio={stream_min_visible_ratio:.2f}"
            )
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")
            door_verifier = None

        mirror_lr = stream_config.get("mirror_left_right", False)

        _has_min = "min_unlock_seconds" in stream_config
        _has_max = "max_unlock_seconds" in stream_config
        stream_min_unlock = float(stream_config["min_unlock_seconds"]) if _has_min else float(config.MIN_UNLOCK_SECONDS)
        stream_max_unlock = float(stream_config["max_unlock_seconds"]) if _has_max else float(config.MAX_UNLOCK_SECONDS)

        stream_evening_second_unlocker_timeout = float(
            stream_config.get("evening_second_unlocker_timeout_seconds", config.EVENING_SECOND_UNLOCKER_TIMEOUT_SECONDS)
        )
        print(
            f"[SYSTEM] Lock interaction window: "
            f"min={stream_min_unlock}s ({'stream' if _has_min else 'global default'}), "
            f"max={stream_max_unlock}s ({'stream' if _has_max else 'global default'})"
        )

        state_machine = DualAuthStateMachine(
            roi_manager, int(fps),
            mirror_left_right=mirror_lr,
            min_unlock_seconds=stream_min_unlock,
            max_unlock_seconds=stream_max_unlock,
        )
        visualizer   = Visualizer()
        alert_system = AlertSystem(evidence_dir=evidence_dir, runtime_logger=runtime_logger)
        frame_quality_gate = FrameQualityGate(
            door_corner_roi=active_rois.get("DOOR_CORNER_ROI"),
            degraded_after_frames=int(getattr(config, "FRAME_QUALITY_DEGRADED_AFTER_FRAMES", 15)),
            recovery_good_frames=int(getattr(config, "FRAME_QUALITY_RECOVERY_GOOD_FRAMES", 5)),
            stale_after_frames=int(getattr(config, "FRAME_QUALITY_STALE_AFTER_FRAMES", 90)),
        )

        startup_ist    = datetime.now(IST)
        last_reset_date = startup_ist.strftime("%Y-%m-%d")
        curr_hm        = startup_ist.strftime("%H:%M")

        if test_window:
            print(f"[SYSTEM] TEST MODE: Forcing {test_window.upper()} window logic.")
            morning_check_done = (test_window == "evening")
            evening_check_done = (test_window == "morning")
        else:
            morning_check_done = curr_hm > "12:00"
            evening_check_done = curr_hm > "23:58"
            if morning_check_done:
                print(f"[SYSTEM] Startup after 12:00 PM IST. Morning check for {last_reset_date} marked as SKIPPED.")
            if evening_check_done:
                print(f"[SYSTEM] Startup after 11:58 PM IST. Evening check for {last_reset_date} marked as SKIPPED.")

        frame_idx                    = 0
        last_processed_frame_idx     = 0
        tracked_persons              = {}
        occupancy_status             = "OK"
        auth_result                  = {
            "authorized":       False,
            "lock_a_authorized": False,
            "lock_b_authorized": False,
            "violation_type":   "INCOMPLETE",
        }
        active_auth_window           = None
        auth_success_logged_by_window = {"morning": False, "evening": False}
        evening_auth_started         = False
        # Wall-clock time of the OPEN->CLOSE instant. The second-unlocker timeout is
        # measured in REAL seconds from this, not frame-time, so quality-freezes (which
        # stop frame_idx advancing) cannot stretch the timeout past the window end.
        # No frame is cached: all captures use the live (quality-gated GOOD) frame at
        # decision time — real-time evidence only, never a stored/stale image.
        evening_closing_time         = None
        # Morning grace: wall-clock deadline after a CLOSED->OPEN that looked unauthorized.
        # While set, the loop keeps processing GOOD frames and re-checking auth before
        # snapshotting, so unlocker IDs can settle (no p1/p2-null race at the bare instant).
        morning_grace_deadline       = None
        last_door_state              = None
        is_door_open                 = False
        ssim_val                     = None
        intensity_val                = None
        intensity_diff               = None
        door_transition_pending      = False
        frame_quality_result         = None
        frame_quality_was_frozen     = False
        video_quality_degraded_logged = False
        tracking_active              = False
        presence_triggered           = False
        _presence_scan_count         = 0
        persons_auth_status          = None
        auth_check_complete          = False
        stream_priority_active       = False
        last_priority_activity_frame = 0
        priority_activity_grace_frames = max(
            1,
            int(float(getattr(config, "PRIORITY_ACTIVITY_GRACE_SECONDS", 2.0)) * fps),
        )

        live_window_available        = can_show_live_window(show_live)

        if live_window_available:
            cv2.namedWindow(f"Two-Man Rule Live ROI Debug - {cam_id}", cv2.WINDOW_NORMAL)

        print("[SYSTEM] Starting frame processing loop...")
        t_loop_start          = time.perf_counter()
        processed_frames_count = 0
        video_read_timeout = (
            float(getattr(config, "RTSP_READ_TIMEOUT_SECONDS", 1.5))
            if total_frames <= 0
            else 2.0
        )


        while True:
            ret, frame, frame_ist = video.read_frame(block=True, timeout=video_read_timeout)
            if not ret:
                if stream_config.get("camera_id"):
                    print(f"[SYSTEM] Stream {stream_config['camera_id']} died. Exiting for restart.")
                break
            if frame is None:
                continue

            clean_frame = frame.copy()
            frame_idx  += 1

            # ===== IST TIME & DAILY RESET =====
            # now_ist is pinned at cap.read() grab time inside VideoHandler.
            # Falls back to wall-clock only if grab timestamp missing.
            now_ist    = frame_ist if frame_ist is not None else datetime.now(IST)
            frame_quality_result = frame_quality_gate.evaluate(frame)
            today_str  = now_ist.strftime("%Y-%m-%d")
            if last_reset_date != today_str:
                print(f"[SYSTEM] Midnight reset for {today_str} IST.")
                morning_check_done            = False
                evening_check_done            = False
                auth_success_logged_by_window = {"morning": False, "evening": False}
                active_auth_window            = None
                evening_auth_started          = False
                presence_triggered            = False
                _presence_scan_count          = 0
                stream_priority_active        = False
                last_priority_activity_frame  = 0
                lkg_detections                = []
                lkg_consecutive_timeouts      = 0
                state_machine.reset_session()
                last_reset_date = today_str

            curr_hour_min = now_ist.strftime("%H:%M")

            if test_window:
                is_morning_window = (test_window == "morning")
                is_evening_window = (test_window == "evening")
            else:
                is_morning_window = "07:00" <= curr_hour_min <= "12:00"
                is_evening_window = "19:00" <= curr_hour_min <= "23:00"

            current_auth_window = None
            if is_morning_window and not morning_check_done:
                current_auth_window = "morning"
            elif is_evening_window and not evening_check_done:
                current_auth_window = "evening"

            if current_auth_window != active_auth_window:
                _transition_ist = now_ist.strftime("%H:%M:%S")
                if current_auth_window is None:
                    if active_auth_window is not None:
                        print(f"[SYSTEM] [{_transition_ist} IST] Leaving {active_auth_window} auth window. Clearing auth session.")
                        runtime_logger.write_event(
                            event_type="WINDOW_CLOSE",
                            message=f"{active_auth_window} auth window closed",
                            level="INFO",
                            details={"window": active_auth_window, "frame_idx": frame_idx},
                            frame_idx=frame_idx,
                            ts_ist=now_ist,
                        )
                        # Window ended with no verdict. Any stream still alive here is
                        # by definition no-verdict — a concluded audit already exited.
                        # Mark the window handled and exit so the detector's VRAM is
                        # released: no stream/model lingers past its window. Reuses the
                        # post-loop completion teardown (mark .done + empty_cache + exit).
                        # Applies to BOTH morning and evening. Test/debug modes exempt —
                        # there the window never "leaves" (forced flags).
                        if not auth_check_complete and not test_window and not debug and not show_all_detections:
                            # If evening was ARMED (door did OPEN->CLOSE) but no verdict
                            # was reached before the window ended, emit the unauthorized
                            # capture now on the live (quality-gated GOOD) frame — so any
                            # stream whose door closed always leaves a witness snapshot
                            # instead of exiting silently. Real-time frame only, never a
                            # stored/stale image. Streams that never armed exit silently.
                            if active_auth_window == "evening" and evening_auth_started:
                                capture(
                                    alert_system, clean_frame,
                                    "DOOR_CLOSE_UNAUTHORIZED_PRESENCE",
                                    evidence_dir=evidence_dir, cam_id=cam_id,
                                    site_name=site_name,
                                    site_id=stream_config.get("site_id", ""),
                                    details={
                                        "authorized": False,
                                        "p1_id": state_machine.session.get("id_a"),
                                        "p2_id": state_machine.session.get("id_b"),
                                        "wait_time": "window_end",
                                        "reason": "second_unlocker_timeout",
                                    },
                                    check_type="Evening", visualizer=visualizer,
                                    unlocker_labels=unlocker_labels,
                                    tracked_persons=tracked_persons,
                                    auth_result=auth_result, is_door_open=is_door_open,
                                    persons_auth_status=False,
                                    runtime_logger=runtime_logger,
                                    frame_idx=frame_idx, frame_ist=now_ist,
                                )
                                print(f"[EVENING] {cam_id}: window ended while armed — "
                                      f"saved closing-frame witness capture.")
                            print(
                                f"[SYSTEM] {cam_id}: {active_auth_window} window ended with no "
                                f"verdict. Releasing VRAM and exiting."
                            )
                            auth_check_complete = True
                            break
                else:
                    print(f"[SYSTEM] [{_transition_ist} IST] Starting {current_auth_window} auth window. Detector already hot. Fresh session.")
                    runtime_logger.write_event(
                        event_type="WINDOW_OPEN",
                        message=f"{current_auth_window} auth window activated — detector hot, inference immediate",
                        level="INFO",
                        details={"window": current_auth_window, "frame_idx": frame_idx, "process_every": process_every},
                        frame_idx=frame_idx,
                        ts_ist=now_ist,
                    )

                state_machine.reset_session()
                evening_auth_started   = False
                presence_triggered     = False
                _presence_scan_count   = 0
                stream_priority_active = False
                last_priority_activity_frame = 0
                tracked_persons        = {}
                occupancy_status       = "OK"
                lkg_detections         = []
                lkg_consecutive_timeouts = 0
                auth_result = {
                    "authorized":        False,
                    "lock_a_authorized": False,
                    "lock_b_authorized": False,
                    "violation_type":    "INCOMPLETE",
                }
                active_auth_window = current_auth_window

            auth_window_open = current_auth_window is not None

            # Outside any auth window: keep RTSP alive, do nothing else
            if not auth_window_open and not debug and not show_all_detections:
                if total_frames <= 0:  # live RTSP: sleep between ticks
                    time.sleep(0.5)
                continue

            auth_tracking_allowed = (
                current_auth_window == "morning"
                or current_auth_window == "evening"
            )

            # Active Monitoring: Run AI continuously during active windows.
            if current_auth_window == "morning" and not morning_check_done:
                scanner_inference_active = True
            elif current_auth_window == "evening" and not evening_check_done:
                scanner_inference_active = True
            else:
                scanner_inference_active = debug or show_all_detections

            tracking_active = scanner_inference_active
            frame_quality_active = auth_window_open or debug or show_all_detections

            if frame_quality_active and frame_quality_result.usable and frame_quality_was_frozen:
                runtime_logger.write_event(
                    event_type="VIDEO_RECOVERED",
                    message="Frame quality recovered; resuming AI/door/auth processing",
                    level="INFO",
                    details={
                        "status": frame_quality_result.status.value,
                        "reason": frame_quality_result.reason,
                        "consecutive_good": frame_quality_result.consecutive_good,
                    },
                    frame_idx=frame_idx,
                    ts_ist=now_ist,
                )
                print(
                    f"[{cam_id}] VIDEO_RECOVERED after "
                    f"{frame_quality_result.consecutive_good} good frame(s)."
                )
                frame_quality_was_frozen = False
                video_quality_degraded_logged = False
                # Fix 2: re-baseline the door after a quality gap so the first
                # post-recovery frame (often a dawn washout) cannot immediately emit a
                # false transition. Door re-settles over a fresh debounce window.
                if door_verifier is not None:
                    door_verifier.reset_stabilization()

            if should_freeze_for_frame_quality(frame_quality_result, frame_quality_active):
                last_processed_frame_idx = frame_idx
                frame_quality_was_frozen = True
                quality_metrics = {
                    k: round(float(v), 3)
                    for k, v in frame_quality_result.metrics.items()
                    if isinstance(v, (int, float, np.integer, np.floating))
                }

                should_log_quality = (
                    frame_quality_result.consecutive_bad == 1
                    or frame_quality_result.consecutive_bad % 30 == 0
                )
                if frame_quality_result.degraded and not video_quality_degraded_logged:
                    should_log_quality = True
                    video_quality_degraded_logged = True
                    quality_event_type = "VIDEO_DEGRADED"
                    quality_level = "WARNING"
                else:
                    quality_event_type = "VIDEO_QUALITY_FREEZE"
                    quality_level = "WARNING"

                if should_log_quality:
                    runtime_logger.write_event(
                        event_type=quality_event_type,
                        message="Frame quality gate froze AI/door/auth processing",
                        level=quality_level,
                        details={
                            "status": frame_quality_result.status.value,
                            "reason": frame_quality_result.reason,
                            "usable": frame_quality_result.usable,
                            "degraded": frame_quality_result.degraded,
                            "consecutive_bad": frame_quality_result.consecutive_bad,
                            "consecutive_good": frame_quality_result.consecutive_good,
                            "metrics": quality_metrics,
                        },
                        frame_idx=frame_idx,
                        ts_ist=now_ist,
                    )
                    print(
                        f"[{cam_id}] {quality_event_type}: "
                        f"{frame_quality_result.status.value} "
                        f"reason={frame_quality_result.reason} "
                        f"bad={frame_quality_result.consecutive_bad} "
                        f"good={frame_quality_result.consecutive_good}"
                    )

                if live_window_available:
                    try:
                        status_text = (
                            f"VIDEO QUALITY: {frame_quality_result.status.value} "
                            f"| {frame_quality_result.reason}"
                        )
                        visualizer.draw_status_text(
                            frame,
                            status_text,
                            (10, 105),
                            color=(0, 165, 255),
                            bg_color=(0, 50, 100),
                        )
                        cv2.imshow(f"Two-Man Rule Live ROI Debug - {cam_id}", frame)
                        wait_ms = max(1, int(1000 / max(fps, 1)))
                        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                            print("[SYSTEM] Live preview stopped by user.")
                            break
                    except cv2.error as e:
                        print(f"[WARNING] Live preview unavailable: {e}")
                        live_window_available = False
                continue

            # ===== PIPELINE =====
            # Auth windows enable scanner inference. Scanner detections promote only
            # this stream to a priority lane; the window alone no longer does that.
            _fsm_active = (
                state_machine.session.get("candidate_a") is not None
                or state_machine.session.get("candidate_b") is not None
                or state_machine.session.get("id_a") is not None
                or state_machine.session.get("id_b") is not None
                or state_machine.session.get("zone_occupied") == True
            )
            _priority_inference_active = scanner_inference_active and auth_tracking_allowed and (
                stream_priority_active or _fsm_active
            )

            should_process_frame = (
                frame_idx == 1
                or (frame_idx - last_processed_frame_idx) >= process_every
            )

            if should_process_frame:
                frame_step = max(frame_idx - last_processed_frame_idx, 1)
                last_processed_frame_idx = frame_idx

                t0 = time.perf_counter()

                if tracking_active:
                    # Timing instrumentation: arrival -> inference start/end -> GPU mem
                    t_arrival = time.perf_counter()
                    gpu_before_mb = None
                    try:
                        import torch
                        if torch.cuda.is_available():
                            gpu_before_mb = float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
                    except Exception:
                        gpu_before_mb = None

                    t_inf_start = time.perf_counter()
                    raw_detections = detector.detect(frame)
                    t_inf_end = time.perf_counter()

                    gpu_after_mb = None
                    try:
                        import torch
                        if torch.cuda.is_available():
                            gpu_after_mb = float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
                    except Exception:
                        gpu_after_mb = None

                    freeze_fsm_for_timeout = False

                    # Record a summary event for this frame
                    try:
                        ft = FrameTimingTracker.instance()
                        num_persons = None
                        avg_conf = None
                        wrist_kps = None
                        if isinstance(raw_detections, list):
                            num_persons = len(raw_detections)
                            confs = []
                            wrist_kps = 0
                            for person in raw_detections:
                                kp = person.get("keypoints") or []
                                # keypoints entries are (x,y,conf)
                                for idx in (config.KEYPOINT_WRIST_LEFT, config.KEYPOINT_WRIST_RIGHT):
                                    if idx < len(kp) and kp[idx][2] > 0.25:
                                        wrist_kps += 1
                                # try gather confidences from keypoints if available
                                for k in kp:
                                    if len(k) >= 3:
                                        confs.append(float(k[2]))
                            if confs:
                                avg_conf = sum(confs) / len(confs)

                        event = FrameTimingEvent(
                            frame_idx=frame_idx,
                            camera_id=cam_id,
                            t_arrival=t_arrival,
                            t_inference_start=t_inf_start,
                            t_inference_end=t_inf_end,
                            gpu_mem_before_mb=gpu_before_mb,
                            gpu_mem_after_mb=gpu_after_mb,
                            num_persons=num_persons,
                            avg_confidence=avg_conf,
                            wrist_keypoints=wrist_kps,
                        )
                        ft.record_event(event)
                    except Exception:
                        pass

                    # ----------------------------------------------------------
                    # Inference result handling — three distinct outcomes:
                    #
                    #   raw_detections is None  → INFERENCE_TIMEOUT
                    #       The GPU server did not respond within the deadline.
                    #       Transport failure — do NOT advance the FSM toward
                    #       UNAUTHORIZED.  Keep tracker/FSM state frozen so
                    #       candidate grace and timers are not wiped.
                    #
                    #   raw_detections == []    → GENUINE EMPTY RESULT
                    #       GPU ran successfully and found no persons.
                    #       Advance FSM normally (persons left the scene).
                    #
                    #   raw_detections is list  → SUCCESSFUL DETECTION
                    #       Normal path. Update LKG cache and reset timeout counter.
                    # ----------------------------------------------------------
                    if raw_detections is None:
                        # INFERENCE_TIMEOUT: hold tracker/FSM state briefly.
                        lkg_consecutive_timeouts += 1
                        if lkg_consecutive_timeouts == 1:
                            print(
                                f"[{cam_id}] Inference timeout on frame {frame_idx}. "
                                f"Using LKG ({len(lkg_detections)} detections). "
                                f"FSM state preserved."
                            )
                        elif lkg_consecutive_timeouts % 10 == 0:
                            print(
                                f"[{cam_id}] {lkg_consecutive_timeouts} consecutive inference timeouts. "
                                f"LKG age: {lkg_consecutive_timeouts} frames."
                            )

                        if lkg_consecutive_timeouts <= LKG_MAX_AGE:
                            # Short transport timeout: keep last tracker/FSM state intact.
                            # A missing GPU reply should not consume candidate grace.
                            freeze_fsm_for_timeout = True
                            detections = lkg_detections
                        else:
                            # Too many consecutive timeouts — GPU server likely down.
                            # Use empty list so tracker ages out tracks gracefully,
                            # but log as DEGRADED rather than UNAUTHORIZED.
                            detections = []
                            print(
                                f"[{cam_id}] DEGRADED: {lkg_consecutive_timeouts} timeouts. "
                                f"Tracker coasting on empty detections. "
                                f"Check GPU server."
                            )
                    else:
                        # Successful inference (empty or non-empty)
                        if lkg_consecutive_timeouts > 0:
                            print(
                                f"[{cam_id}] Inference restored after "
                                f"{lkg_consecutive_timeouts} timeouts."
                            )
                        lkg_consecutive_timeouts = 0
                        lkg_detections           = raw_detections   # update cache
                        detections               = raw_detections

                    if not freeze_fsm_for_timeout:
                        # Protect verified/candidate IDs from tracker Re-ID theft
                        _verified_ids = set()
                        for _s in ("a", "b"):
                            for _key in (f"id_{_s}", f"candidate_{_s}"):
                                _v = state_machine.session.get(_key)
                                if _v is not None:
                                    _verified_ids.add(_v)
                            _tag = f"P{1 if _s == 'a' else 2}_unlocker"
                            _verified_ids.update(state_machine.all_unlocker_ids.get(_tag, set()))

                        reid_embeddings = reid_extractor.embed(
                            clean_frame, [d["bbox"] for d in detections]
                        ) if detections else None
                        tracked_persons = tracker.update(
                            detections, protected_ids=_verified_ids, embeddings=reid_embeddings
                        )

                        auth_active = (
                            current_auth_window == "morning"
                            # Evening: detection is GATED on the OPEN->CLOSE transition.
                            # Until the door closes (evening_auth_started), no unlocker
                            # candidate/slot/tag/timer is touched — pure scan only.
                            or (current_auth_window == "evening" and evening_auth_started)
                            or (current_auth_window is None and (debug or show_all_detections))
                        )

                        if auth_active:
                            occupancy_status = state_machine.update_occupancy(tracked_persons, frame_step=frame_step)
                            state_machine.update_timers(tracked_persons, frame_step=frame_step, frame=clean_frame)
                            auth_result = state_machine.check_authorization()
                        else:
                            state_machine.active_ids_in_zone = set()
                            state_machine.session["improper_positioning"] = None
                            occupancy_status = "OK"
                            auth_result = {
                                "authorized":        False,
                                "lock_a_authorized": False,
                                "lock_b_authorized": False,
                                "violation_type":    "INCOMPLETE",
                            }

                        priority_activity_now = state_machine.has_priority_activity(tracked_persons)
                        _fsm_active_after = (
                            state_machine.session.get("candidate_a") is not None
                            or state_machine.session.get("candidate_b") is not None
                            or state_machine.session.get("id_a") is not None
                            or state_machine.session.get("id_b") is not None
                            or state_machine.session.get("zone_occupied") == True
                        )
                        if auth_tracking_allowed and (priority_activity_now or _fsm_active_after):
                            if not stream_priority_active:
                                print(f"[{cam_id}] Scanner promoted stream to PRIORITY inference.")
                            stream_priority_active = True
                            last_priority_activity_frame = frame_idx
                        elif stream_priority_active and not _fsm_active_after:
                            inactive_frames = frame_idx - last_priority_activity_frame
                            if inactive_frames >= priority_activity_grace_frames:
                                stream_priority_active = False
                                print(
                                    f"[{cam_id}] Priority activity idle for "
                                    f"{inactive_frames / max(fps, 1):.1f}s. Returning to SCANNER."
                                )

                inference_ms = (time.perf_counter() - t0) * 1000.0
                processed_frames_count += 1

                if frame_idx % 150 == 0:
                    elapsed    = time.perf_counter() - t_loop_start
                    actual_fps = processed_frames_count / elapsed if elapsed > 0 else 0
                    telemetry  = video.get_telemetry()
                    timeout_tag = f" | LKG timeouts: {lkg_consecutive_timeouts}" if lkg_consecutive_timeouts else ""
                    print(
                        f"[METRICS] {cam_id} | FPS: {actual_fps:.1f} | AI: {inference_ms:.1f}ms | "
                        f"Reconnects: {telemetry['reconnect_count']}{timeout_tag}"
                    )
                    t_loop_start           = time.perf_counter()
                    processed_frames_count = 0

            # ===== DOOR VERIFICATION (every frame during active audit) =====
            # Decoupled from the pose-inference throttle (`should_process_frame`)
            # so a CLOSED<->OPEN transition is detected within one frame (+the
            # debounce), instead of lagging by up to `process_every` frames. This
            # keeps the captured snapshot frame and its timestamp in sync with the
            # real door movement and with the unlocker-activity timeline. SSIM on
            # the small DOOR_CORNER patch is cheap, so per-frame cost is minimal.
            if door_verifier and (tracking_active or debug):
                if current_auth_window == "morning" and not morning_check_done:
                    check_door = True
                elif current_auth_window == "evening" and not evening_check_done:
                    check_door = True
                else:
                    check_door = state_machine.should_check_door_state()

                check_door = check_door or debug
                if check_door:
                    is_door_open = door_verifier.verify(frame, tracked_persons=tracked_persons, ts_ist=now_ist)
                else:
                    is_door_open = last_door_state if last_door_state is not None else False

                ssim_val        = door_verifier.get_last_ssim()
                intensity_val   = door_verifier.get_last_intensity()
                intensity_diff  = door_verifier.get_last_intensity_diff()
                door_transition_pending = door_verifier.is_transition_pending()
            elif not door_verifier:
                door_transition_pending = False

            # ===== VISUALIZATION =====
            draw_rois(visualizer, frame, active_rois)

            unlocker_labels = get_unlocker_labels(state_machine, tracked_persons, frame=frame)
            _show_all       = show_all_detections or debug

            for track_id, person in tracked_persons.items():
                if track_id in unlocker_labels:
                    label = unlocker_labels[track_id]
                elif _show_all:
                    label = f"ID {track_id}"
                else:
                    continue

                if auth_result["authorized"] and track_id in unlocker_labels:
                    color = config.COLOR_AUTHORIZED
                elif track_id in unlocker_labels:
                    color = config.COLOR_UNLOCKING
                else:
                    color = config.COLOR_DETECTED
                visualizer.draw_bounding_box(frame, person["bbox"], color, label)

            visible_pose_ids = set(tracked_persons.keys()) if (debug or show_all_detections) else set(unlocker_labels)
            draw_pose_debug(frame, tracked_persons, visible_pose_ids)
            draw_lost_verified_ghosts(visualizer, frame, state_machine, unlocker_labels)

            locks_center = roi_manager.get_roi_center("LOCKS_ROI")
            if locks_center:
                id_a_done = state_machine.session.get("id_a") is not None
                id_b_done = state_machine.session.get("id_b") is not None
                pct_a = min((state_machine.session.get("timer_a_seconds", 0) / stream_min_unlock) * 100, 100)
                pct_b = min((state_machine.session.get("timer_b_seconds", 0) / stream_min_unlock) * 100, 100)

                if not id_a_done:
                    if pct_a > 0:
                        visualizer.draw_circular_progress_bar(frame, tuple(map(int, locks_center)), pct_a)
                elif not id_b_done:
                    if pct_b > 0:
                        visualizer.draw_circular_progress_bar(frame, tuple(map(int, locks_center)), pct_b)
                else:
                    visualizer.draw_circular_progress_bar(frame, tuple(map(int, locks_center)), 100)

            n = 0
            if tracking_active:
                if state_machine.session.get("candidate_a") is not None or state_machine.session.get("id_a") is not None:
                    n += 1
                if state_machine.session.get("candidate_b") is not None or state_machine.session.get("id_b") is not None:
                    n += 1

            # Surface LKG timeout status in the HUD
            timeout_hud = (
                f" | LKG:{lkg_consecutive_timeouts}f"
                if lkg_consecutive_timeouts > 0
                else ""
            )
            auth_status_text  = auth_result["authorized"] if tracking_active else "OFF"
            state_status_text = state_machine.session["sequence_state"] if tracking_active else "IDLE_OUTSIDE_AUDIT"
            visualizer.draw_status_text(
                frame,
                f"Unlockers: {n} | State: {state_status_text} | Auth: {auth_status_text}{timeout_hud}",
                (10, 30),
            )

            if should_process_frame:
                if lkg_consecutive_timeouts > 0:
                    ai_status = (
                        f"AI: TIMEOUT (LKG age {lkg_consecutive_timeouts}f) | "
                        f"Every {process_every} frame(s)"
                    )
                    visualizer.draw_status_text(frame, ai_status, (10, 55), color=(0, 165, 255))
                else:
                    if not scanner_inference_active:
                        ai_status = (
                            "AI tracking: waiting for OPEN->CLOSE"
                            if current_auth_window == "evening"
                            else "AI tracking: OFF outside audit window"
                        )
                    elif current_auth_window == "evening" and not evening_auth_started:
                        ai_status = (
                            f"AI SCANNER: {inference_ms:.0f}ms | "
                            "watching for OPEN->CLOSE"
                        )
                    elif _priority_inference_active:
                        ai_status = (
                            f"AI PRIORITY: {inference_ms:.0f}ms | Every 1 frame | "
                            f"IDs only for unlockers"
                        )
                    else:
                        ai_status = (
                            f"AI SCANNER: {inference_ms:.0f}ms | Every {process_every} frame(s)"
                        )
                    visualizer.draw_status_text(frame, ai_status, (10, 55))

            door_status_label = "--" if door_transition_pending else ("OPEN" if is_door_open else "CLOSED")
            if ssim_val is not None:
                if debug and intensity_val is not None:
                    visualizer.draw_status_text(
                        frame,
                        f"SSIM: {ssim_val:.3f} | Intensity: {intensity_val:.1f} | Door: {door_status_label}",
                        (10, 80),
                    )
                else:
                    visualizer.draw_status_text(frame, f"SSIM: {ssim_val:.3f} | Door: {door_status_label}", (10, 80))

            door_status_text = f"DOOR: {door_status_label}"
            door_color = (
                (200, 200, 200) if door_transition_pending
                else ((0, 0, 255) if is_door_open else (0, 255, 0))
            )
            cv2.putText(frame, door_status_text, (frame.shape[1] - 300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, door_status_text, (frame.shape[1] - 300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, door_color, 2, cv2.LINE_AA)

            # ===== EVENTS + CAPTURE =====
            site_id = stream_config.get("site_id", "")

            def _capture(event_type, details, check_type="System", frame_override=None):
                # Defaults to the live frame at event-trigger time. frame_override lets
                # the evening timeout use the cached door-closing frame (witness of who
                # closed the door) instead of an empty late live frame.
                cap_frame = frame_override if frame_override is not None else clean_frame
                capture(
                    alert_system, cap_frame, event_type,
                    evidence_dir=evidence_dir,
                    cam_id=cam_id,
                    site_name=site_name,
                    site_id=site_id,
                    details=details or {},
                    check_type=check_type,
                    visualizer=visualizer,
                    unlocker_labels=unlocker_labels,
                    tracked_persons=tracked_persons,
                    auth_result=auth_result,
                    is_door_open=is_door_open,
                    persons_auth_status=persons_auth_status,
                    runtime_logger=runtime_logger,
                    frame_idx=frame_idx,
                    frame_ist=now_ist,
                )

            if tracking_active and should_process_frame and occupancy_status == "VIOLATION_OVERCROWD":
                visualizer.draw_status_text(
                    frame, "SECURITY BREACH: Unauthorized Presence",
                    (10, 80), color=(0, 0, 255), bg_color=(0, 0, 100),
                )

            if tracking_active and should_process_frame and auth_result.get("violation_type") == "SAME_ID":
                visualizer.draw_status_text(
                    frame, "SECURITY BREACH: SAME PERSON ATTEMPTING DUAL UNLOCK",
                    (10, 80), color=(0, 0, 255), bg_color=(0, 0, 100),
                )
                if "SAME_ID" not in state_machine.session["captured_violations"]:
                    state_machine.session["captured_violations"].append("SAME_ID")
                    if current_auth_window != "morning":
                        persons_auth_status = False
                        # Label by LIVE door state — evening same-person violations fire
                        # post-close, so the capture must not hard-label DOOR_OPEN.
                        same_id_event_type = (
                            "DOOR_OPEN_UNAUTHORIZED_PRESENCE" if is_door_open
                            else "DOOR_CLOSE_UNAUTHORIZED_PRESENCE"
                        )
                        _capture(
                            same_id_event_type,
                            {
                                "authorized": False,
                                "p1_id": state_machine.session.get("id_a")
                                         or state_machine.session.get("same_id_offender"),
                                "p2_id": state_machine.session.get("id_b"),
                                "reason": "same_person_tried_both_slots",
                            },
                            current_auth_window or "Security",
                        )

                if current_auth_window == "evening":
                    evening_check_done   = True
                    evening_auth_started = False
                    stream_priority_active = False
                    print("[EVENING] Dual Auth FAILED: Same person attempted both unlocks.")
                    state_machine.session["violation_type"] = None
                    auth_check_complete = True
                    break
                elif current_auth_window == "morning":
                    if is_door_open:
                        persons_auth_status = False
                        _capture(
                            "DOOR_OPEN_UNAUTHORIZED_PRESENCE",
                            {"reason": "same_person_tried_both_slots"},
                            "Morning",
                        )
                        morning_check_done = True
                        stream_priority_active = False
                        print("[MORNING] Dual Auth FAILED: Same person attempted both unlocks (Door open). Exiting.")
                        state_machine.session["violation_type"] = None
                        auth_check_complete = True
                        break
                    else:
                        print("[MORNING] SAME_ID violation detected. Will capture at CLOSED->OPEN transition.")

                state_machine.session["violation_type"] = None

            if tracking_active and should_process_frame and state_machine.session.get("improper_positioning"):
                bad_id    = state_machine.session["improper_positioning"]
                bad_label = unlocker_labels.get(bad_id, "ignored detection")
                visualizer.draw_status_text(
                    frame, f"IMPROPER POSITIONING: {bad_label}",
                    (10, 105), color=(0, 165, 255), bg_color=(0, 50, 100),
                )

            door_transition = None
            if door_verifier and not getattr(door_verifier, "has_stabilized", True):
                last_door_state = None
            else:
                if last_door_state is not None and last_door_state != is_door_open:
                    door_transition = "CLOSED_TO_OPEN" if is_door_open else "OPEN_TO_CLOSED"
                last_door_state = is_door_open

            # ===== MORNING CHECK =====
            if is_morning_window and not morning_check_done:
                # A new CLOSED->OPEN, or grace already running, both drive the verdict.
                _morning_open_event = (
                    door_transition == "CLOSED_TO_OPEN" and morning_grace_deadline is None
                ) or (morning_grace_deadline is not None)
                if _morning_open_event:
                    is_auth = auth_result["authorized"]
                    both_in_interaction = state_machine.verified_unlockers_in_interaction_zone(tracked_persons)
                    same_id_now = "SAME_ID" in state_machine.session.get("captured_violations", [])
                    grace_expired = (
                        morning_grace_deadline is not None and now_ist >= morning_grace_deadline
                    )

                    if same_id_now:
                        persons_auth_status = False
                        _capture("DOOR_OPEN_UNAUTHORIZED_PRESENCE", {
                            "authorized": False,
                            "p1_id": state_machine.session.get("id_a")
                                     or state_machine.session.get("same_id_offender"),
                            "p2_id": state_machine.session.get("id_b"),
                            "transition": "CLOSED_TO_OPEN",
                            "both_in_interaction_zone": both_in_interaction,
                            "reason": "same_person_tried_both_slots",
                        }, "Morning")
                        print(f"[MORNING] UNAUTHORIZED CLOSED->OPEN at {curr_hour_min} IST (SAME_ID).")
                        state_machine.session["door_open_captured"] = True
                        morning_grace_deadline = None
                        morning_check_done = True
                        stream_priority_active = False
                        auth_check_complete = True
                        break
                    elif is_auth and both_in_interaction:
                        persons_auth_status = True
                        _capture("DOOR_OPEN_AUTHORIZED_PRESENCE", {
                            "authorized": True,
                            "p1_id": state_machine.session.get("id_a"),
                            "p2_id": state_machine.session.get("id_b"),
                            "transition": "CLOSED_TO_OPEN",
                            "both_in_interaction_zone": True,
                        }, "Morning")
                        print(f"[MORNING] Authorized CLOSED->OPEN confirmed at {curr_hour_min} IST.")
                        state_machine.session["door_open_captured"] = True
                        morning_grace_deadline = None
                        morning_check_done = True
                        stream_priority_active = False
                        auth_check_complete = True
                        break
                    elif morning_grace_deadline is None:
                        # First unauthorized-looking instant — start the grace and WAIT for
                        # unlocker IDs to settle before snapshotting (no bare-instant null).
                        morning_grace_deadline = now_ist + timedelta(
                            seconds=config.MORNING_CAPTURE_GRACE_SECONDS
                        )
                        print(f"[MORNING] CLOSED->OPEN at {curr_hour_min} IST — "
                              f"grace {config.MORNING_CAPTURE_GRACE_SECONDS:.1f}s for unlocker IDs.")
                        visualizer.draw_status_text(
                            frame, "MORNING CHECK: CONFIRMING UNLOCKERS...",
                            (10, 130), color=(0, 255, 255), bg_color=(0, 50, 50),
                        )
                    elif not is_door_open:
                        # Door fell back to CLOSED during grace — the open was not sustained
                        # (spurious/flicker). Abort: no capture, wait for a real transition.
                        morning_grace_deadline = None
                        print(f"[MORNING] Open not sustained during grace at {curr_hour_min} IST — aborting capture.")
                    elif not grace_expired:
                        # Still within grace, still unauthorized — keep waiting on GOOD frames.
                        visualizer.draw_status_text(
                            frame, "MORNING CHECK: CONFIRMING UNLOCKERS...",
                            (10, 130), color=(0, 255, 255), bg_color=(0, 50, 50),
                        )
                    else:
                        persons_auth_status = False
                        _capture("DOOR_OPEN_UNAUTHORIZED_PRESENCE", {
                            "authorized": False,
                            "p1_id": state_machine.session.get("id_a"),
                            "p2_id": state_machine.session.get("id_b"),
                            "transition": "CLOSED_TO_OPEN",
                            "both_in_interaction_zone": both_in_interaction,
                            "reason": "missing_dual_auth_or_interaction_zone",
                        }, "Morning")
                        print(f"[MORNING] UNAUTHORIZED CLOSED->OPEN at {curr_hour_min} IST.")
                        state_machine.session["door_open_captured"] = True
                        morning_grace_deadline = None
                        morning_check_done = True
                        stream_priority_active = False
                        auth_check_complete = True
                        break

                elif not morning_check_done:
                    # Idle display while waiting for door transition
                    if door_transition_pending:
                        visualizer.draw_status_text(
                            frame, "STATUS: DOOR STATE STABILIZING...",
                            (10, 130), color=(0, 255, 255), bg_color=(0, 50, 50),
                        )
                    elif auth_result["authorized"]:
                        visualizer.draw_status_text(
                            frame, "MORNING CHECK: 2 UNLOCKERS READY - WAITING FOR CLOSED->OPEN",
                            (10, 130), color=(0, 255, 255), bg_color=(0, 50, 50),
                        )
                    else:
                        visualizer.draw_status_text(
                            frame, "MORNING CHECK: IDENTIFYING 2 UNLOCKERS",
                            (10, 130), color=(0, 165, 255),
                        )

            # ===== EVENING CHECK =====
            elif is_evening_window and not evening_check_done:
                if door_transition == "OPEN_TO_CLOSED" and not evening_auth_started:
                    # HARD RESET at the OPEN->CLOSE transition: unlocker detection must
                    # start 100% fresh. Nothing accumulated while the door was open may
                    # survive — no pre-close candidate, slot, tag, or timer carries over.
                    state_machine.reset_session()
                    evening_auth_started = True
                    state_machine.session["door_closing_start_frame"] = frame_idx
                    evening_closing_time = now_ist
                    print(f"[EVENING] Door OPEN->CLOSE detected at {curr_hour_min} IST. Starting unlocker check.")

                if evening_auth_started:
                    if (
                        "door_closing_start_frame" not in state_machine.session
                        or state_machine.session["door_closing_start_frame"] is None
                    ):
                        state_machine.session["door_closing_start_frame"] = frame_idx
                    if evening_closing_time is None:
                        evening_closing_time = now_ist

                    # WALL-CLOCK timeout: real seconds since the door closed, so
                    # quality-freezes (which halt frame_idx) cannot stretch it past
                    # the window end. Frame marker kept for legacy/debug only.
                    elapsed_seconds = (now_ist - evening_closing_time).total_seconds()
                    is_auth         = auth_result["authorized"]

                    if is_auth:
                        persons_auth_status = True
                        _capture("DOOR_CLOSE_AUTHORIZED_PRESENCE", {
                            "authorized": True,
                            "p1_id":      state_machine.session.get("id_a"),
                            "p2_id":      state_machine.session.get("id_b"),
                            "wait_time":  f"{elapsed_seconds:.1f}s",
                        }, "Evening")
                        print(f"[EVENING] Authorized closure confirmed at {curr_hour_min} IST.")
                        evening_check_done   = True
                        evening_auth_started = False
                        stream_priority_active = False
                        auth_check_complete = True
                        break
                    elif elapsed_seconds >= stream_evening_second_unlocker_timeout:
                        persons_auth_status = False
                        # Capture on the live (quality-gated GOOD) frame at the timeout
                        # instant — real-time evidence only, never a stored/stale image.
                        _capture("DOOR_CLOSE_UNAUTHORIZED_PRESENCE", {
                            "authorized": False,
                            "p1_id":      state_machine.session.get("id_a"),
                            "p2_id":      state_machine.session.get("id_b"),
                            "wait_time":  f"{elapsed_seconds:.1f}s Timeout",
                            "reason":     "second_unlocker_timeout",
                        }, "Evening")
                        print(f"[EVENING] UNAUTHORIZED closure (timeout) at {curr_hour_min} IST.")
                        evening_check_done   = True
                        evening_auth_started = False
                        stream_priority_active = False
                        auth_check_complete = True
                        break
                    else:
                        wait_time_rem = stream_evening_second_unlocker_timeout - elapsed_seconds
                        visualizer.draw_status_text(
                            frame, f"EVENING CHECK: WAITING FOR 2 UNLOCKERS ({wait_time_rem:.0f}s)",
                            (10, 130), color=(0, 255, 255), bg_color=(0, 50, 50),
                        )
                else:
                    if door_transition_pending:
                        visualizer.draw_status_text(
                            frame, "STATUS: DOOR STATE STABILIZING...",
                            (10, 130), color=(0, 255, 255), bg_color=(0, 50, 50),
                        )
                    else:
                        visualizer.draw_status_text(
                            frame, "STATUS: EVENING WINDOW OPEN - WATCHING FOR OPEN->CLOSE",
                            (10, 130), color=(0, 165, 255),
                        )

            else:
                if not is_morning_window and not is_evening_window:
                    status_msg = f"STATUS: IDLE | NEXT WINDOW: {'MORNING' if curr_hour_min < '06:00' else 'EVENING'}"
                elif morning_check_done and is_morning_window:
                    status_msg = "STATUS: MORNING CHECK COMPLETE"
                elif evening_check_done and is_evening_window:
                    status_msg = "STATUS: EVENING CHECK COMPLETE"
                else:
                    status_msg = "STATUS: SYSTEM IDLE (OUTSIDE WINDOWS)"
                visualizer.draw_status_text(frame, status_msg, (10, 130), color=(200, 200, 200))

            if not is_door_open:
                state_machine.session["door_open_captured"]    = False
                state_machine.session["door_opening_start_frame"] = None
            else:
                state_machine.session["door_closing_start_frame"] = None

            if (
                tracking_active
                and should_process_frame
                and auth_result["authorized"]
                and not auth_success_logged_by_window.get(active_auth_window, False)
            ):
                auth_success_logged_by_window[active_auth_window] = True
                runtime_logger.write_event(
                    event_type="DUAL_AUTH_SUCCESS",
                    message=f"Dual person authorization confirmed for {active_auth_window} window",
                    level="INFO",
                    details={
                        "window": active_auth_window,
                        "p1_id":  state_machine.session.get("id_a"),
                        "p2_id":  state_machine.session.get("id_b"),
                    },
                    frame_idx=frame_idx,
                    ts_ist=now_ist,
                )
                print(f"[SYSTEM] Dual person authorization confirmed for {active_auth_window} window.")

            # ===== PROGRESS LOG =====
            if (tracking_active or debug) and frame_idx % 30 == 0:
                timers  = (
                    f"P1:{state_machine.session['timer_a_seconds']:.1f}s "
                    f"P2:{state_machine.session['timer_b_seconds']:.1f}s"
                )
                cand_a  = f"ID {state_machine.session['candidate_a']}" if state_machine.session["candidate_a"] is not None else "-"
                cand_b  = f"ID {state_machine.session['candidate_b']}" if state_machine.session["candidate_b"] is not None else "-"
                id_a    = f"ID {state_machine.session['id_a']}"        if state_machine.session["id_a"] is not None else "-"
                id_b    = f"ID {state_machine.session['id_b']}"        if state_machine.session["id_b"] is not None else "-"
                ssim_str = f" | Door SSIM: {ssim_val:.3f}" if ssim_val is not None else ""
                intensity_str = ""
                if debug and ssim_val is not None and intensity_val is not None:
                    intensity_str = f" | Intensity: {intensity_val:.1f} (Δ{intensity_diff:.1f})"
                timeout_str = (
                    f" | LKG timeouts: {lkg_consecutive_timeouts}"
                    if lkg_consecutive_timeouts > 0 else ""
                )
                total_frames_val = total_frames if total_frames > 0 else "LIVE"
                progress_val     = f"({video.get_progress():.1f}%)" if total_frames > 0 else ""
                print(
                    f"[PROGRESS] Frame {frame_idx}/{total_frames_val} {progress_val} "
                    f"| Unlockers: {n} | State: {state_machine.session['sequence_state']} "
                    f"| Candidates: P1={cand_a} P2={cand_b} "
                    f"| Verified: P1={id_a} P2={id_b} | {timers}"
                    f"{ssim_str}{intensity_str}{timeout_str}"
                )

            if live_window_available:
                try:
                    if debug:
                        display_frame = frame
                    else:
                        display_frame = clean_frame.copy()
                        visualizer.draw_client_overlays(
                            display_frame, unlocker_labels, tracked_persons,
                            auth_result, is_door_open,
                            persons_auth_status=persons_auth_status,
                        )
                    cv2.imshow(f"Two-Man Rule Live ROI Debug - {cam_id}", display_frame)
                    wait_ms = max(1, int(1000 / max(fps, 1)))
                    if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                        print("[SYSTEM] Live preview stopped by user.")
                        break
                except cv2.error as e:
                    print(f"[WARNING] Live preview unavailable: {e}")
                    live_window_available = False

    print("[SYSTEM] Processing complete.")
    try:
        ft_path = f"logs/frame_timing_{cam_id}.csv"
        FrameTimingTracker.instance().export_csv(ft_path)
        print(f"[SYSTEM] Frame timing CSV exported: {ft_path}")
    except Exception:
        pass
    print(f"[SYSTEM] Evidence files: {len(os.listdir(evidence_dir))}")
    if 'live_window_available' in locals() and live_window_available:
        cv2.destroyAllWindows()

    if auth_check_complete:
        # Persist a durable completion marker BEFORE exiting. A respawn that lands
        # inside the same window reads this and sleeps to the next window instead
        # of re-auditing and emitting duplicate captures. Skipped in test/debug
        # modes so manual reruns are not suppressed.
        if not test_window and not debug and not show_all_detections:
            _completed_window = active_auth_window or _current_window_name()
            _completed_date = datetime.now(IST).strftime("%Y-%m-%d")
            _mark_window_complete(cam_id, _completed_date, _completed_window)

        # Hard-release GPU by exiting the process. empty_cache() does NOT free
        # the CUDA context, cuDNN workspace, or cuBLAS handles — only process
        # exit does (driver tears down ctx). Supervisor / outer loop respawns
        # a fresh child near the next window, which re-enters the pre-window
        # sleep above and holds zero VRAM until then.
        detector = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        sleep_secs = _seconds_until_next_window()
        print(
            f"[SYSTEM] Auth check complete. Exiting to fully release CUDA context. "
            f"Next window in {sleep_secs:.0f}s; supervisor will respawn."
        )
        sys.exit(0)

    return auth_check_complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Two-Man Rule monitoring with live ROI overlay.")
    parser.add_argument("--stream-index", type=int, default=None)
    parser.add_argument("--stream-indices", type=str, default=None,
                        help="Comma-separated stream indexes (e.g. 0,2,4).")
    parser.add_argument(
        "--stream-video",
        action="append",
        default=[],
        metavar="INDEX=VIDEO_PATH",
        help="Override one configured stream with a local video. Repeat for batches.",
    )
    parser.add_argument("video_source", nargs="?", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--scale-rois", action="store_true")
    parser.add_argument("--process-every", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--show-all-detections", action="store_true")
    parser.add_argument("--test-window", type=str, choices=["morning", "evening"], default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--shm-slot", type=int, default=-1)
    args = parser.parse_args()

    if args.stream_indices is not None and args.stream_index is not None:
        print("[ERROR] Use either --stream-index or --stream-indices, not both.")
        sys.exit(1)

    if args.video_source is not None and args.stream_video:
        print("[ERROR] Use either positional video_source or --stream-video overrides, not both.")
        sys.exit(1)

    try:
        stream_video_sources = _parse_stream_video_overrides(
            args.stream_video,
            len(config.STREAMS_CONFIG),
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    if args.stream_indices is not None:
        if args.video_source is not None:
            print("[ERROR] video_source override is only supported with a single stream.")
            sys.exit(1)
        try:
            selected_stream_indexes = _parse_stream_indices(args.stream_indices, len(config.STREAMS_CONFIG))
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)
    elif args.stream_index is not None:
        selected_stream_indexes = [args.stream_index]
    elif args.video_source is not None:
        selected_stream_indexes = [0]
    elif stream_video_sources:
        selected_stream_indexes = list(stream_video_sources)
    else:
        selected_stream_indexes = list(range(len(config.STREAMS_CONFIG)))

    unselected_video_indexes = set(stream_video_sources) - set(selected_stream_indexes)
    if unselected_video_indexes:
        extras = ",".join(str(idx) for idx in sorted(unselected_video_indexes))
        print(f"[ERROR] --stream-video overrides are not selected by stream index: {extras}.")
        sys.exit(1)

    if len(selected_stream_indexes) > 1:
        if len(selected_stream_indexes) == len(config.STREAMS_CONFIG):
            print(f"[SYSTEM] Launching all {len(config.STREAMS_CONFIG)} streams in parallel...")
        else:
            print(f"[SYSTEM] Launching selected streams: {selected_stream_indexes}")

        max_streams_per_gpu = max(int(getattr(config, "MAX_STREAMS_PER_GPU", 1)), 1)
        extra_launch_delay = float(getattr(config, "EXTRA_STREAM_LAUNCH_DELAY_SECONDS", 0.0))
        if len(selected_stream_indexes) > max_streams_per_gpu:
            print(
                f"[SYSTEM] Over the soft cap: {len(selected_stream_indexes)} streams > "
                f"MAX_STREAMS_PER_GPU={max_streams_per_gpu}. Throughput depends on GPU headroom."
            )

        processes = []
        base_cmd  = [sys.executable, sys.argv[0]]
        if args.show:
            print("[WARNING] --show adds GPU/CPU overhead; not recommended for production.")
            base_cmd.append("--show")
        if args.scale_rois:          base_cmd.append("--scale-rois")
        base_cmd.extend(["--process-every", str(args.process_every)])
        base_cmd.extend(["--device", args.device])
        if args.no_half:             base_cmd.append("--no-half")
        if args.show_all_detections: base_cmd.append("--show-all-detections")
        if args.test_window:         base_cmd.extend(["--test-window", args.test_window])
        if args.debug:               base_cmd.append("--debug")

        for pos, i in enumerate(selected_stream_indexes):
            cmd = base_cmd + ["--stream-index", str(i)]
            stream_video_source = stream_video_sources.get(i)
            if stream_video_source is not None:
                cmd.append(stream_video_source)
            p = _start_stream_process(cmd)
            processes.append({
                "process": p,
                "cmd": cmd,
                "stream_index": i,
                "restart_on_exit": _video_source_is_live(stream_video_source),
                "completed": False,
            })
            print(f"[SYSTEM] Launched Stream {i} (PID: {p.pid})")

            if pos < len(selected_stream_indexes) - 1:
                delay = getattr(config, "STAGGER_START_DELAY", 2.0)
                if pos + 1 >= max_streams_per_gpu:
                    delay = max(delay, extra_launch_delay)
                print(f"[SYSTEM] Waiting {delay}s before next launch...")
                time.sleep(delay)

        print("[SYSTEM] All streams launched. Supervisor active.")
        try:
            while True:
                time.sleep(5)
                for entry in processes:
                    if entry["completed"]:
                        continue
                    p = entry["process"]
                    if p.poll() is not None:
                        s_idx = entry["stream_index"]
                        if not entry["restart_on_exit"]:
                            entry["completed"] = True
                            print(
                                f"[SYSTEM] Video Stream {s_idx} finished "
                                f"(PID: {p.pid}, code {p.returncode})."
                            )
                            continue

                        print(
                            f"[WATCHDOG] Stream {s_idx} (PID: {p.pid}) died "
                            f"(code {p.returncode}). Restarting..."
                        )
                        new_p = _start_stream_process(entry["cmd"])
                        entry["process"] = new_p
                        print(f"[WATCHDOG] Stream {s_idx} restarted (PID: {new_p.pid})")
                if all(entry["completed"] for entry in processes):
                    print("[SYSTEM] All local video streams finished.")
                    break
        except KeyboardInterrupt:
            print("\n[SYSTEM] Shutting down...")
        finally:
            previous_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                for entry in processes:
                    _terminate_stream_process(entry["process"])
                # No shared GPU masters to shutdown in standalone mode
            finally:
                signal.signal(signal.SIGINT, previous_sigint_handler)
        sys.exit(0)

    # ---- Single stream path ----
    args.stream_index = selected_stream_indexes[0]
    if args.stream_index < 0 or args.stream_index >= len(config.STREAMS_CONFIG):
        print(f"[ERROR] Invalid stream-index {args.stream_index}.")
        sys.exit(1)

    stream_config = config.STREAMS_CONFIG[args.stream_index]
    if args.shm_slot >= 0:
        stream_config['shm_slot'] = args.shm_slot

    video_source = stream_video_sources.get(args.stream_index, args.video_source)
    if video_source is not None and video_source.isdigit():
        video_source = int(video_source)

    if args.test_window is None:
        args.test_window = _infer_test_window_from_video_source(video_source)
        if args.test_window is not None:
            print(
                f"[SYSTEM] Inferred test window '{args.test_window}' "
                f"from video source '{video_source}'."
            )

    # Single-stream invocation: run one window cycle then exit. Supervisor (or
    # systemd / cron) is responsible for respawn. main() handles its own
    # pre-window sleep and calls sys.exit(0) on auth completion, which would
    # break a Python-level `while True` anyway, so no restart loop here.
    _restore_terminal_capture = enable_terminal_capture(
        base_dir=config.BASE_LOG_DIR,
        site_name=stream_config["site_name"],
        camera_id=stream_config["camera_id"],
    )
    exit_code = 0
    try:
        try:
            main(
                stream_config=stream_config,
                video_source=video_source,
                show_live=args.show,
                scale_rois=args.scale_rois,
                process_every=args.process_every,
                device=args.device,
                half=not args.no_half,
                show_all_detections=args.show_all_detections,
                test_window=args.test_window,
                debug=args.debug,
            )
        except KeyboardInterrupt:
            print("\n[SYSTEM] Interrupted by user. Exiting.")
        except SystemExit:
            raise
        except Exception as exc:
            print(f"[SYSTEM] Unhandled exception in main(): {exc}")
            import traceback; traceback.print_exc()
            exit_code = 1
    finally:
        _restore_terminal_capture()
    sys.exit(exit_code)
