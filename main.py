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
from models.pose_detector import PoseDetector, _InferenceServer, start_inference_manager, get_shared_queues
from models.tracker import PersonTracker
from models.door_verifier import DoorVerifier
from logic.roi_manager import ROIManager
from logic.state_machine import DualAuthStateMachine

from io_.video_handler import VideoHandler
from io_.visualizer import Visualizer
from io_.alert_system import AlertSystem
from io_.runtime_logger import RuntimeEventLogger
from io_.terminal_tee import enable_terminal_capture
import config
import json

# Global reference for cleanup in single-stream mode
detector = None

def _gpu_server_entry(req_q, response_registry, shm_name,
                      base_log_dir, device, half, is_priority):
    """Redirect GPU server stdout/stderr to the standard daily logs.
    Must be at module level for multiprocessing pickle to work."""
    from models.pose_detector import _InferenceServer
    from io_.terminal_tee import enable_terminal_capture
    tier_name = "priority" if is_priority else "scan"
    restore_logs = enable_terminal_capture(
        base_dir=base_log_dir,
        site_name="gpu_servers",
        camera_id=".",
        file_prefix=f"gpu_{tier_name}"
    )
    try:
        _srv = _InferenceServer(device=device, half=half)
        _srv.run(req_q, response_registry, shm_name, is_priority=is_priority)
    finally:
        restore_logs()


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
    height_ref_bbox = (getattr(state_machine, "slot_height_ref", {}).get(slot)) or ref_bbox
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
):
    """Unified capture + log helper. Saves annotated frame + paired JSON metadata."""
    now_ist  = datetime.now(IST)
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

    alert_system.log_event(event_type, details or {})
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
    process_every: int = 3,
    device: str = "auto",
    half: bool = True,
    show_all_detections: bool = False,
    test_window: str = None,
    debug: bool = False,
    shared_inference: bool = False,
):
    global detector
    video_source = video_source or stream_config["rtsp_url"]
    cam_id       = stream_config["camera_id"]
    site_name    = stream_config["site_name"]
    evidence_dir = os.path.join(config.BASE_OUTPUT_DIR, site_name, cam_id)
    runtime_logger = RuntimeEventLogger(
        base_dir=config.BASE_LOG_DIR,
        site_name=site_name,
        camera_id=cam_id,
    )

    print(f"[SYSTEM] Initializing Two-Man Rule Monitoring System for {site_name} - {cam_id}...")
    _alert_counter[0] = 0
    os.makedirs(evidence_dir, exist_ok=True)
    runtime_logger.write_event(
        event_type="STREAM_START",
        message="Stream worker initialized",
        level="INFO",
        details={
            "video_source": str(video_source),
            "evidence_dir": evidence_dir,
            "log_file":     runtime_logger.current_file_path,
        },
    )

    detector = PoseDetector(device=device, half=half, shared_mode=shared_inference)
    if shared_inference:
        scan_q, priority_q, res_registry, pri_registry, shm_config = get_shared_queues()
        if scan_q is not None and res_registry is not None:
            # Inject assigned SHM slot into registry slot_map if provided via CLI
            if shm_config and 'shm_slot' in stream_config:
                slot_map = dict(shm_config.get('slot_map', {}))
                slot_map[detector.client_id] = stream_config['shm_slot']
                shm_config['slot_map'] = slot_map

            # set_queues creates the private response queue and registers it
            detector.set_queues(scan_q, priority_q, res_registry, pri_registry, shm_config=shm_config)
        else:
            print("[SYSTEM] Shared mode failed to connect. Falling back to standalone.")
            detector.shared_mode = False

    tracker      = PersonTracker()
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

            door_verifier = DoorVerifier(
                stream_config["closed_door_reference"],
                door_corner_roi=active_rois["DOOR_CORNER_ROI"],
                similarity_threshold=stream_ssim_thresh,
                debounce_threshold=stream_debounce,
                intensity_threshold=stream_intensity_thresh,
                motion_threshold=stream_motion_thresh,
                darkening_protection=bool(stream_darkening),
                min_visible_ratio=stream_min_visible_ratio,
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

        startup_ist    = datetime.now(IST)
        last_reset_date = startup_ist.strftime("%Y-%m-%d")
        curr_hm        = startup_ist.strftime("%H:%M")

        if test_window:
            print(f"[SYSTEM] TEST MODE: Forcing {test_window.upper()} window logic.")
            morning_check_done = (test_window == "evening")
            evening_check_done = (test_window == "morning")
        else:
            morning_check_done = curr_hm > "11:00"
            evening_check_done = curr_hm > "23:58"
            if morning_check_done:
                print(f"[SYSTEM] Startup after 11:00 AM IST. Morning check for {last_reset_date} marked as SKIPPED.")
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
        last_door_state              = None
        is_door_open                 = False
        ssim_val                     = None
        intensity_val                = None
        intensity_diff               = None
        door_transition_pending      = False
        tracking_active              = False
        persons_auth_status          = None

        live_window_available        = can_show_live_window(show_live)

        if live_window_available:
            cv2.namedWindow(f"Two-Man Rule Live ROI Debug - {cam_id}", cv2.WINDOW_NORMAL)

        print("[SYSTEM] Starting frame processing loop...")
        t_loop_start          = time.perf_counter()
        processed_frames_count = 0

        while True:
            ret, frame = video.read_frame(block=True, timeout=0.1)
            if not ret:
                if stream_config.get("camera_id"):
                    print(f"[SYSTEM] Stream {stream_config['camera_id']} died. Exiting for restart.")
                break
            if frame is None:
                continue

            clean_frame = frame.copy()
            frame_idx  += 1

            # ===== IST TIME & DAILY RESET =====
            now_ist    = datetime.now(IST)
            today_str  = now_ist.strftime("%Y-%m-%d")
            if last_reset_date != today_str:
                print(f"[SYSTEM] Midnight reset for {today_str} IST.")
                morning_check_done            = False
                evening_check_done            = False
                auth_success_logged_by_window = {"morning": False, "evening": False}
                active_auth_window            = None
                evening_auth_started          = False
                lkg_detections                = []
                lkg_consecutive_timeouts      = 0
                state_machine.reset_session()
                last_reset_date = today_str

            curr_hour_min = now_ist.strftime("%H:%M")

            if test_window:
                is_morning_window = (test_window == "morning")
                is_evening_window = (test_window == "evening")
            else:
                is_morning_window = "07:00" <= curr_hour_min <= "11:00"
                is_evening_window = "19:00" <= curr_hour_min <= "23:00"

            current_auth_window = None
            if is_morning_window and not morning_check_done:
                current_auth_window = "morning"
            elif is_evening_window and not evening_check_done:
                current_auth_window = "evening"

            if current_auth_window != active_auth_window:
                if current_auth_window is None:
                    if active_auth_window is not None:
                        print(f"[SYSTEM] Leaving {active_auth_window} auth window. Clearing auth session.")
                else:
                    print(f"[SYSTEM] Starting {current_auth_window} auth window with a fresh auth session.")

                state_machine.reset_session()
                evening_auth_started   = False
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

            tracking_active = (
                current_auth_window == "morning"
                or (current_auth_window == "evening" and evening_auth_started)
                or debug
                or show_all_detections
            )

            # ===== PIPELINE =====
            # FSM activity signal — computed locally from this stream's own session state.
            # True when candidate_a/b or verified id_a/b is set (unlocker interaction active).
            # This is passed to detector.detect() so it can self-route to the priority
            # inference queue for faster GPU turnaround during the critical auth window.
            _fsm_active = tracking_active and (
                state_machine.session.get("candidate_a") is not None
                or state_machine.session.get("candidate_b") is not None
                or state_machine.session.get("id_a") is not None
                or state_machine.session.get("id_b") is not None
                or state_machine.session.get("zone_occupied") == True
            )

            # Dynamic process_every: run at full rate when an unlocker is being tracked.
            # Falls back to CLI-specified process_every for idle/scanning streams.
            _effective_process_every = (
                1
                if (_fsm_active and shared_inference)
                else process_every
            )

            should_process_frame = (
                frame_idx == 1
                or (frame_idx - last_processed_frame_idx) >= _effective_process_every
            )

            if should_process_frame:
                frame_step = max(frame_idx - last_processed_frame_idx, 1)
                last_processed_frame_idx = frame_idx

                t0 = time.perf_counter()

                if tracking_active:
                    raw_detections = detector.detect(frame, fsm_is_active=_fsm_active)

                    # ----------------------------------------------------------
                    # Inference result handling — three distinct outcomes:
                    #
                    #   raw_detections is None  → INFERENCE_TIMEOUT
                    #       The GPU server did not respond within the deadline.
                    #       Transport failure — do NOT advance the FSM toward
                    #       UNAUTHORIZED.  Reuse LKG detections so the tracker
                    #       can coast and timers are not wiped.
                    #
                    #   raw_detections == []    → GENUINE EMPTY RESULT
                    #       GPU ran successfully and found no persons.
                    #       Advance FSM normally (persons left the scene).
                    #
                    #   raw_detections is list  → SUCCESSFUL DETECTION
                    #       Normal path. Update LKG cache and reset timeout counter.
                    # ----------------------------------------------------------
                    if raw_detections is None:
                        # INFERENCE_TIMEOUT — hold state with LKG
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
                            # Use stale-but-valid detections to coast the tracker
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

                    # Protect verified/candidate IDs from tracker Re-ID theft
                    _verified_ids = set()
                    for _s in ("a", "b"):
                        for _key in (f"id_{_s}", f"candidate_{_s}"):
                            _v = state_machine.session.get(_key)
                            if _v is not None:
                                _verified_ids.add(_v)
                        _tag = f"P{1 if _s == 'a' else 2}_unlocker"
                        _verified_ids.update(state_machine.all_unlocker_ids.get(_tag, set()))

                    tracked_persons = tracker.update(detections, protected_ids=_verified_ids)

                    auth_active = (
                        current_auth_window == "morning"
                        or (current_auth_window == "evening" and evening_auth_started)
                        or (current_auth_window is None and (debug or show_all_detections))
                    )

                    if auth_active:
                        occupancy_status = state_machine.update_occupancy(tracked_persons, frame_step=frame_step)
                        state_machine.update_timers(tracked_persons, frame_step=frame_step)
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

                is_door_open = False
                ssim_val     = None
                if door_verifier:
                    if current_auth_window == "morning" and not morning_check_done:
                        check_door = (
                            state_machine.session.get("id_a") is not None
                            or state_machine.session.get("id_b") is not None
                            or state_machine.session.get("candidate_a") is not None
                            or state_machine.session.get("candidate_b") is not None
                        )
                    elif current_auth_window == "evening" and not evening_check_done:
                        check_door = True
                    else:
                        check_door = state_machine.should_check_door_state()

                    check_door = check_door or debug
                    if check_door:
                        is_door_open = door_verifier.verify(frame, tracked_persons=tracked_persons)
                    else:
                        is_door_open = last_door_state if last_door_state is not None else False

                    ssim_val        = door_verifier.get_last_ssim()
                    intensity_val   = door_verifier.get_last_intensity()
                    intensity_diff  = door_verifier.get_last_intensity_diff()
                    door_transition_pending = door_verifier.is_transition_pending()
                else:
                    door_transition_pending = False

                inference_ms = (time.perf_counter() - t0) * 1000.0
                processed_frames_count += 1

                if frame_idx % 150 == 0:
                    elapsed    = time.perf_counter() - t_loop_start
                    actual_fps = processed_frames_count / elapsed if elapsed > 0 else 0
                    telemetry  = video.get_telemetry()
                    timeout_tag = f" | LKG timeouts: {lkg_consecutive_timeouts}" if lkg_consecutive_timeouts else ""
                    print(
                        f"[METRICS] {cam_id} | FPS: {actual_fps:.1f} | AI: {inference_ms:.1f}ms | "
                        f"Queue Delay: {telemetry['queue_delay_ms']:.1f}ms | "
                        f"Drops: {telemetry['dropped_frames']}{timeout_tag}"
                    )
                    t_loop_start           = time.perf_counter()
                    processed_frames_count = 0

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
                    if not tracking_active:
                        ai_status = (
                            "AI tracking: waiting for OPEN->CLOSE"
                            if current_auth_window == "evening"
                            else "AI tracking: OFF outside audit window"
                        )
                    else:
                        ai_status = (
                            f"AI: {inference_ms:.0f}ms | Every {process_every} frame(s) | "
                            f"IDs only for unlockers"
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

            def _capture(event_type, details, check_type="System"):
                capture(
                    alert_system, clean_frame, event_type,
                    evidence_dir=evidence_dir,
                    cam_id=cam_id,
                    site_name=site_name,
                    site_id=site_id,
                    details=details,
                    check_type=check_type,
                    visualizer=visualizer,
                    unlocker_labels=unlocker_labels,
                    tracked_persons=tracked_persons,
                    auth_result=auth_result,
                    is_door_open=is_door_open,
                    persons_auth_status=persons_auth_status,
                    runtime_logger=runtime_logger,
                    frame_idx=frame_idx,
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
                    persons_auth_status = False
                    _capture(
                        "DOOR_OPEN_UNAUTHORIZED_PRESENCE",
                        {"reason": "same_person_tried_both_slots"},
                        current_auth_window or "Security",
                    )
                    state_machine.session["captured_violations"].append("SAME_ID")

                if current_auth_window == "evening":
                    evening_check_done   = True
                    evening_auth_started = False
                    print("[EVENING] Dual Auth FAILED: Same person attempted both unlocks.")
                elif current_auth_window == "morning":
                    if is_door_open:
                        morning_check_done = True
                        print("[MORNING] Dual Auth FAILED: Same person attempted both unlocks (Door open). Exiting.")
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
            if last_door_state is not None and last_door_state != is_door_open:
                door_transition = "CLOSED_TO_OPEN" if is_door_open else "OPEN_TO_CLOSED"
            last_door_state = is_door_open

            # ===== MORNING CHECK =====
            if is_morning_window and not morning_check_done:
                if door_transition == "CLOSED_TO_OPEN":
                    is_auth = auth_result["authorized"]
                    both_in_interaction = state_machine.verified_unlockers_in_interaction_zone(tracked_persons)

                    if is_auth and both_in_interaction:
                        persons_auth_status = True
                        _capture("DOOR_OPEN_AUTHORIZED_PRESENCE", {
                            "authorized": True,
                            "p1_id": state_machine.session.get("id_a"),
                            "p2_id": state_machine.session.get("id_b"),
                            "transition": "CLOSED_TO_OPEN",
                            "both_in_interaction_zone": True,
                        }, "Morning")
                        print(f"[MORNING] Authorized CLOSED->OPEN confirmed at {curr_hour_min} IST.")
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
                    morning_check_done = True

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
                    state_machine.reset_session()
                    evening_auth_started = True
                    tracking_active      = True
                    state_machine.session["door_closing_start_frame"] = frame_idx
                    print(f"[EVENING] Door OPEN->CLOSE detected at {curr_hour_min} IST. Starting unlocker check.")

                if evening_auth_started:
                    if (
                        "door_closing_start_frame" not in state_machine.session
                        or state_machine.session["door_closing_start_frame"] is None
                    ):
                        state_machine.session["door_closing_start_frame"] = frame_idx

                    elapsed_frames  = frame_idx - state_machine.session["door_closing_start_frame"]
                    elapsed_seconds = elapsed_frames / fps
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
                    elif elapsed_seconds >= stream_evening_second_unlocker_timeout:
                        persons_auth_status = False
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
                alert_system.log_event("DUAL_AUTH_SUCCESS", {
                    "window": active_auth_window,
                    "p1_id":  state_machine.session.get("id_a"),
                    "p2_id":  state_machine.session.get("id_b"),
                })
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
                )
                print(f"[SYSTEM] Dual person authorization confirmed for {active_auth_window} window.")

            # ===== TEST WINDOW EXIT =====
            if test_window:
                check_done = (
                    (test_window == "morning" and morning_check_done)
                    or (test_window == "evening" and evening_check_done)
                )
                if check_done:
                    print(f"[SYSTEM] Test window '{test_window}' check complete. Exiting.")
                    break

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
    print(f"[SYSTEM] Evidence files: {len(os.listdir(evidence_dir))}")
    if 'live_window_available' in locals() and live_window_available:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Two-Man Rule monitoring with live ROI overlay.")
    parser.add_argument("--stream-index", type=int, default=None)
    parser.add_argument("--stream-indices", type=str, default=None,
                        help="Comma-separated stream indexes (e.g. 0,2,4).")
    parser.add_argument("video_source", nargs="?", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--scale-rois", action="store_true")
    parser.add_argument("--process-every", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--show-all-detections", action="store_true")
    parser.add_argument("--test-window", type=str, choices=["morning", "evening"], default=None)
    parser.add_argument(
        "--shared-inference", action="store_true",
        default=getattr(config, "SHARED_INFERENCE_ENABLED", False),
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--shm-slot", type=int, default=-1)
    args = parser.parse_args()

    if args.stream_indices is not None and args.stream_index is not None:
        print("[ERROR] Use either --stream-index or --stream-indices, not both.")
        sys.exit(1)

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
    else:
        selected_stream_indexes = list(range(len(config.STREAMS_CONFIG)))

    if len(selected_stream_indexes) > 1:
        import subprocess
        if len(selected_stream_indexes) == len(config.STREAMS_CONFIG):
            print(f"[SYSTEM] Launching all {len(config.STREAMS_CONFIG)} streams in parallel...")
        else:
            print(f"[SYSTEM] Launching selected streams: {selected_stream_indexes}")

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

        if args.shared_inference:
            base_cmd.append("--shared-inference")
            print("[SYSTEM] Shared Inference mode enabled. Initialising GPU Masters...")

            from multiprocessing.shared_memory import SharedMemory
            shm_size = getattr(config, "MAX_SHARED_MEMORY_MB", 2048) * 1024 * 1024
            try:
                shm = SharedMemory(create=True, size=shm_size)
                print(f"[SYSTEM] Created {shm_size // (1024*1024)}MB SharedMemory: {shm.name}")
            except Exception as e:
                print(f"[SYSTEM] WARNING: Could not create SharedMemory ({e}). Falling back to queue copy.")
                shm = None

            manager           = mp.Manager()
            req_q_scan        = manager.Queue()   # Scanning tier: idle/background streams
            req_q_priority    = manager.Queue()   # Priority tier: active unlocker streams
            response_registry = manager.dict()    # {client_id: private mp.Queue}
            priority_registry = manager.dict()    # {client_id: "SCANNING"|"ACTIVE"} (observability)

            shm_config = manager.dict({
                'name':      shm.name if shm else None,
                'slot_size': shm_size // len(getattr(config, "STREAMS_CONFIG", [1] * 10)),
                'slot_map':  {},
            })

            manager_server = start_inference_manager(
                req_q_scan, req_q_priority,
                response_registry, priority_registry,
                shm_config,
            )
            threading.Thread(target=manager_server.serve_forever, daemon=True).start()

            gpu_log_dir = getattr(config, "BASE_LOG_DIR", ".")

            shm_name = shm.name if shm else None

            # SCANNING server — handles all idle/background streams
            scanning_proc = mp.Process(
                target=_gpu_server_entry,
                args=(req_q_scan, response_registry, shm_name,
                      gpu_log_dir, args.device, not args.no_half, False),
                daemon=True,
            )
            scanning_proc.start()
            print(f"[SYSTEM] GPU SCANNING server active (PID: {scanning_proc.pid}). "
                  f"Logs in: {os.path.join(gpu_log_dir, 'gpu_master', 'scan')}")

            # PRIORITY server — handles active-unlocker streams
            priority_proc = mp.Process(
                target=_gpu_server_entry,
                args=(req_q_priority, response_registry, shm_name,
                      gpu_log_dir, args.device, not args.no_half, True),
                daemon=True,
            )
            priority_proc.start()
            print(f"[SYSTEM] GPU PRIORITY server active (PID: {priority_proc.pid}). "
                  f"Logs in: {os.path.join(gpu_log_dir, 'gpu_master', 'priority')}")

            print(
                f"[SYSTEM] Dual GPU Masters active. "
                f"Per-client routing enabled for {len(selected_stream_indexes)} streams."
            )

        for pos, i in enumerate(selected_stream_indexes):
            cmd = base_cmd + ["--stream-index", str(i)]
            if args.shared_inference and shm:
                cmd.extend(["--shm-slot", str(pos)])
            p = subprocess.Popen(cmd)
            processes.append((p, cmd, i))
            print(f"[SYSTEM] Launched Stream {i} (PID: {p.pid})")

            if pos < len(selected_stream_indexes) - 1:
                delay = getattr(config, "STAGGER_START_DELAY", 2.0)
                print(f"[SYSTEM] Waiting {delay}s before next launch...")
                time.sleep(delay)

        print("[SYSTEM] All streams launched. Supervisor active.")
        try:
            while True:
                time.sleep(5)
                for idx, (p, cmd, s_idx) in enumerate(processes):
                    if p.poll() is not None:
                        print(
                            f"[WATCHDOG] Stream {s_idx} (PID: {p.pid}) died "
                            f"(code {p.returncode}). Restarting..."
                        )
                        new_p = subprocess.Popen(cmd)
                        processes[idx] = (new_p, cmd, s_idx)
                        print(f"[WATCHDOG] Stream {s_idx} restarted (PID: {new_p.pid})")
        except KeyboardInterrupt:
            print("\n[SYSTEM] Shutting down...")
            for p, _, _ in processes:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
            if args.shared_inference:
                print("[SYSTEM] Shutting down GPU Masters...")
                for _proc_name, _proc in [("scanning", locals().get("scanning_proc")),
                                          ("priority",  locals().get("priority_proc"))]:
                    if _proc is not None:
                        _proc.terminate()
                        _proc.join(timeout=2)
                if 'shm' in locals() and shm:
                    shm.close()
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass
        sys.exit(0)

    # ---- Single stream path ----
    args.stream_index = selected_stream_indexes[0]
    if args.stream_index < 0 or args.stream_index >= len(config.STREAMS_CONFIG):
        print(f"[ERROR] Invalid stream-index {args.stream_index}.")
        sys.exit(1)

    stream_config = config.STREAMS_CONFIG[args.stream_index]
    if args.shm_slot >= 0:
        stream_config['shm_slot'] = args.shm_slot

    video_source = args.video_source
    if video_source is not None and video_source.isdigit():
        video_source = int(video_source)

    _is_live = video_source is None or (
        isinstance(video_source, str) and video_source.lower().startswith("rtsp://")
    )

    while True:
        _restore_terminal_capture = enable_terminal_capture(
            base_dir=config.BASE_LOG_DIR,
            site_name=stream_config["site_name"],
            camera_id=stream_config["camera_id"],
        )
        should_sleep_before_restart = False
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
                    shared_inference=args.shared_inference,
                )
            except KeyboardInterrupt:
                print("\n[SYSTEM] Interrupted by user. Exiting.")
                break
            except Exception as exc:
                print(f"[SYSTEM] Unhandled exception in main(): {exc}")
                import traceback; traceback.print_exc()

            if not _is_live:
                break

            print("[SYSTEM] Restarting stream in 10 seconds...")
            should_sleep_before_restart = True
        finally:
            if args.shared_inference and detector:
                detector.cleanup()
            _restore_terminal_capture()

        if should_sleep_before_restart:
            time.sleep(10)