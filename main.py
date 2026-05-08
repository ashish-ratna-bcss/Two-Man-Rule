# main.py
import sys
from datetime import datetime, timezone, timedelta
IST = timezone(timedelta(hours=5, minutes=30))
import os
import cv2
import numpy as np
import argparse
import time
import math
from models.pose_detector import PoseDetector
from models.tracker import PersonTracker
from models.door_verifier import DoorVerifier
from logic.roi_manager import ROIManager
from logic.state_machine import DualAuthStateMachine

from io_.video_handler import VideoHandler
from io_.visualizer import Visualizer
from io_.alert_system import AlertSystem
import config
import json


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
        "LOCKS_ROI": transform(stream_rois["LOCKS_ROI"]),
        "DOOR_ROI": transform(stream_rois["DOOR_ROI"]),
        "STANDING_ZONE": transform(stream_rois["STANDING_ZONE"]),
        "INTERACTION_ZONE": transform(stream_rois["INTERACTION_ZONE"]),
        "DOOR_CORNER_ROI": transform(stream_rois["DOOR_CORNER_ROI"]),
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
        "STANDING_ZONE": ((0, 200, 255), 1),
        "LOCKS_ROI": ((0, 255, 255), 2),
        "DOOR_ROI": ((255, 0, 0), 1),
        "DOOR_CORNER_ROI": ((255, 255, 255), 2),
    }
    for name, points in rois.items():
        color, thickness = roi_styles[name]
        visualizer.draw_roi_polygon(frame, points, color, thickness)
        visualizer.draw_roi_label(frame, name, points, color)


def _bbox_height(bbox) -> float:
    """Return bbox pixel height."""
    if bbox is None or len(bbox) < 4:
        return 0.0
    return float(bbox[3] - bbox[1])


def _bbox_size_matches(ref_bbox, candidate_bbox, tolerance: float = 0.4) -> bool:
    """True if candidate height is within `tolerance` fraction of reference height."""
    ref_h = _bbox_height(ref_bbox)
    if ref_h <= 0:
        return True  # No reference — don't reject
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
    """Apply verified-unlocker label with multi-factor ReID.

    Priority order:
    1. Primary ID still tracked — direct.
    2. Any previously-tagged alt-ID still tracked — identity already confirmed.
    3. Anchor + size fallback — when tracker ID switches.
    """
    session = state_machine.session
    primary_id = session.get(f"id_{slot}")
    if primary_id is None:
        return

    primary_id = int(primary_id)
    tag = f"P{1 if slot == 'a' else 2}_unlocker"
    other_tag = f"P{2 if slot == 'a' else 1}_unlocker"

    def _is_other_unlocker(tid):
        return state_machine.unlocker_tags.get(tid) == other_tag

    # 1. Primary ID directly tracked
    if primary_id in tracked_persons:
        labels[primary_id] = f"{slot_label} ID {primary_id}"
        return

    # 2. Any tagged alt-ID still tracked
    for alt_id in state_machine.get_all_ids_for_tag(tag):
        if alt_id in tracked_persons and alt_id not in labels:
            labels[alt_id] = f"{slot_label} (alt ID {alt_id})"
            print(f"[VIZ] {slot_label} alt-ID {alt_id} (primary={primary_id})")
            return

    # Build unlabelled, non-other-unlocker candidates
    candidates = {
        tid: p for tid, p in tracked_persons.items()
        if tid not in labels and not _is_other_unlocker(tid)
    }
    if not candidates:
        return

    # 3. Live-origin + size fallback
    anchor = state_machine.verified_anchors.get(slot)
    ref_bbox = state_machine.last_seen_bbox.get(slot)
    height_ref_bbox = (getattr(state_machine, "slot_height_ref", {}).get(slot)) or ref_bbox
    if anchor is None:
        return

    # Use last_seen_bbox center-bottom as live search origin — follows person movement
    # verified_anchor stays frozen at lock-area; useless after P2 enters the room
    if ref_bbox is not None:
        search_origin = ((ref_bbox[0] + ref_bbox[2]) / 2, float(ref_bbox[3]))
    else:
        search_origin = anchor

    # Grow threshold as person is lost longer — catches post-occlusion re-detection
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
    """Return labels for active candidates and verified unlockers."""
    session = state_machine.session
    labels = {}
    tracked_persons = tracked_persons or {}

    if session.get("candidate_a") is not None:
        track_id = int(session["candidate_a"])
        labels[track_id] = f"P1 unlocking ID {track_id}"
    if session.get("candidate_b") is not None:
        track_id = int(session["candidate_b"])
        labels[track_id] = f"P2 unlocking ID {track_id}"

    _label_verified_slot("a", "P1 verified", state_machine, tracked_persons, labels, frame)
    _label_verified_slot("b", "P2 verified", state_machine, tracked_persons, labels, frame)

    return labels


def draw_lost_verified_ghosts(visualizer: Visualizer, frame: np.ndarray, state_machine: DualAuthStateMachine, unlocker_labels: dict):
    """Draw ghost anchors for verified persons who are currently lost/unassigned."""
    for slot in ("a", "b"):
        if state_machine.session.get(f"id_{slot}") is None:
            anchor = state_machine.verified_anchors.get(slot)
            if anchor is not None:
                # This person was verified but is currently lost
                label = f"RECOVERING P{1 if slot == 'a' else 2}..."
                visualizer.draw_ghost_anchor(frame, anchor, label)


def draw_pose_debug(frame: np.ndarray, tracked_persons: dict, visible_ids: set):
    keypoint_styles = {
        0: ("HEAD", (255, 255, 0)),
        config.KEYPOINT_WRIST_LEFT: ("LW", (0, 255, 0)),
        config.KEYPOINT_WRIST_RIGHT: ("RW", (0, 255, 0)),
        config.KEYPOINT_ELBOW_LEFT: ("LE", (0, 165, 255)),
        config.KEYPOINT_ELBOW_RIGHT: ("RE", (0, 165, 255)),
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
            cv2.putText(frame, label, (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def can_show_live_window(show_live: bool) -> bool:
    if not show_live:
        return False
    if os.name == "posix" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("[WARNING] No display server detected; live window disabled. ROI preview image will still be saved.")
        return False
    return True


def capture(
    alert_system: AlertSystem,
    clean_frame: np.ndarray,
    event_type: str,
    evidence_dir: str,
    cam_id: str,
    site_id: str = "",
    details: dict = None,
    check_type: str = "System",
    visualizer: "Visualizer" = None,
    unlocker_labels: dict = None,
    tracked_persons: dict = None,
    auth_result: dict = None,
    is_door_open: bool = False,
    persons_auth_status=None,
):
    """Unified capture + log helper. Saves annotated frame + paired JSON metadata."""
    now_ist = datetime.now(IST)
    date_str = now_ist.strftime("%d-%m-%Y")
    time_str = now_ist.strftime("%H-%M-%S")

    target_dir = os.path.join(evidence_dir, date_str)
    os.makedirs(target_dir, exist_ok=True)

    _alert_counter[0] += 1
    filename = f"alert_{site_id}_{cam_id}_{date_str}_{time_str}.png"
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

    # Save paired JSON metadata
    json_path = full_path.rsplit('.', 1)[0] + '.json'
    event_data = {
        "site_id": site_id,
        "cam_id": cam_id,
        "window": check_type.lower(),
        "events": [
            {
                "timestamp": now_ist.isoformat(),
                "event_type": event_type,
                "details": details or {},
            }
        ]
    }
    try:
        with open(json_path, "w") as f:
            json.dump(event_data, f, indent=4, cls=_NumpySafeEncoder)
        json_ok = True
    except Exception as e:
        print(f"[ERROR] Failed to save JSON metadata: {e}")
        json_ok = False

    alert_system.log_event(event_type, details or {})
    print(f"[CAPTURE] {event_type}: {full_path} (image={'OK' if ok else 'FAILED'}, json={'OK' if json_ok else 'FAILED'})")
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
):
    video_source = video_source or stream_config["rtsp_url"]
    cam_id = stream_config["camera_id"]
    site_name = stream_config["site_name"]
    evidence_dir = os.path.join(config.BASE_OUTPUT_DIR, site_name, cam_id)

    print(f"[SYSTEM] Initializing Two-Man Rule Monitoring System for {site_name} - {cam_id}...")
    _alert_counter[0] = 0
    os.makedirs(evidence_dir, exist_ok=True)


    detector = PoseDetector(device=device, half=half)
    tracker = PersonTracker()

    roi_manager = ROIManager()

    with VideoHandler(video_source) as video:
        fps = video.get_fps()
        width, height = video.get_dimensions()
        total_frames = video.get_total_frames()
        process_every = max(int(process_every), 1)
        print(f"[VIDEO] FPS: {fps:.1f}, Resolution: {width}x{height}, Total Frames: {total_frames}")
        print("[VIDEO] Processing original frames without resizing.")
        print(f"[VIDEO] Pose inference every {process_every} frame(s). Use --process-every 1 for full analysis.")

        active_rois = setup_rois(roi_manager, stream_config["rois"], width, height, scale_rois=scale_rois)
        print_roi_coordinates(active_rois, width, height, scale_rois)
        try:
            # Read per-stream tuning (fall back to safe production defaults)
            stream_ssim_thresh = float(stream_config.get("ssim_threshold", config.SSIM_THRESHOLD))
            stream_intensity_thresh = stream_config.get("intensity_threshold", None)
            stream_motion_thresh = stream_config.get("motion_threshold", None)
            stream_debounce = int(stream_config.get("debounce_threshold", config.DOOR_DEBOUNCE_FRAMES))

            # Basic validation / clamping to avoid accidental misconfiguration
            stream_ssim_thresh = min(max(stream_ssim_thresh, 0.5), 0.99)
            if stream_intensity_thresh is not None:
                stream_intensity_thresh = float(max(stream_intensity_thresh, 0.0))
            if stream_motion_thresh is not None:
                stream_motion_thresh = float(max(stream_motion_thresh, 0.0))
            stream_debounce = int(min(max(stream_debounce, 1), 600))

            stream_darkening = stream_config.get("darkening_protection", config.DOOR_DARKENING_PROTECTION)

            door_verifier = DoorVerifier(
                stream_config["closed_door_reference"],
                door_corner_roi=active_rois["DOOR_CORNER_ROI"],
                similarity_threshold=stream_ssim_thresh,
                debounce_threshold=stream_debounce,
                intensity_threshold=stream_intensity_thresh,
                motion_threshold=stream_motion_thresh,
                darkening_protection=bool(stream_darkening),
            )
            print(f"[SYSTEM] Door verifier loaded with threshold {stream_ssim_thresh}")
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")
            door_verifier = None

        mirror_lr = stream_config.get("mirror_left_right", False)

        # Per-stream unlock timing — stream value takes full priority over global default
        _has_min = "min_unlock_seconds" in stream_config
        _has_max = "max_unlock_seconds" in stream_config
        stream_min_unlock = float(stream_config["min_unlock_seconds"]) if _has_min else float(config.MIN_UNLOCK_SECONDS)
        stream_max_unlock = float(stream_config["max_unlock_seconds"]) if _has_max else float(config.MAX_UNLOCK_SECONDS)
        stream_morning_post_open_auth = float(stream_config.get("morning_post_open_auth_seconds", config.MORNING_POST_OPEN_AUTH_SECONDS))
        stream_evening_second_unlocker_timeout = float(stream_config.get("evening_second_unlocker_timeout_seconds", config.EVENING_SECOND_UNLOCKER_TIMEOUT_SECONDS))
        print(f"[SYSTEM] Lock interaction window: "
              f"min={stream_min_unlock}s ({'stream' if _has_min else 'global default'}), "
              f"max={stream_max_unlock}s ({'stream' if _has_max else 'global default'})")

        state_machine = DualAuthStateMachine(
            roi_manager, int(fps),
            mirror_left_right=mirror_lr,
            min_unlock_seconds=stream_min_unlock,
            max_unlock_seconds=stream_max_unlock,
        )
        visualizer = Visualizer()
        alert_system = AlertSystem(evidence_dir=evidence_dir)

        # Determine initial state based on current IST time
        startup_ist = datetime.now(IST)
        last_reset_date = startup_ist.strftime("%Y-%m-%d")
        curr_hm = startup_ist.strftime("%H:%M")
        
        # Test mode override
        if test_window:
            print(f"[SYSTEM] TEST MODE: Forcing {test_window.upper()} window logic.")
            morning_check_done = (test_window == "evening")
            evening_check_done = (test_window == "morning")
        else:
            # If started after 11:00 AM, morning check is neglected for today
            morning_check_done = curr_hm > "11:00"
            # If started after 11:58 PM, evening check is neglected for today
            evening_check_done = curr_hm > "23:58"
            
            if morning_check_done:
                print(f"[SYSTEM] Startup after 11:00 AM IST. Morning check for {last_reset_date} marked as SKIPPED.")
            if evening_check_done:
                print(f"[SYSTEM] Startup after 11:58 PM IST. Evening check for {last_reset_date} marked as SKIPPED.")
        
        frame_idx = 0
        debug_frame_saved = False
        roi_preview_saved = False
        last_processed_frame_idx = 0
        tracked_persons = {}
        occupancy_status = "OK"
        auth_result = {
            "authorized": False,
            "lock_a_authorized": False,
            "lock_b_authorized": False,
            "violation_type": "INCOMPLETE",
        }
        active_auth_window = None
        auth_success_logged_by_window = {
            "morning": False,
            "evening": False,
        }
        morning_initial_door_checked = False
        evening_auth_started = False
        last_door_state = None  # To detect transitions
        is_door_open = False
        ssim_val = None
        door_transition_pending = False
        tracking_active = False
        persons_auth_status = None  # None=blank, True=Available, False=Unavailable
        morning_post_open_started = False   # True after CLOSED→OPEN confirmed
        morning_post_open_start_frame = None  # frame_idx when post-open window began
        live_window_available = can_show_live_window(show_live)
        if live_window_available:
            cv2.namedWindow(f"Two-Man Rule Live ROI Debug - {cam_id}", cv2.WINDOW_NORMAL)

        print("[SYSTEM] Starting frame processing loop...")
        t_loop_start = time.perf_counter()
        processed_frames_count = 0

        while True:
            # Wait for latest frame from background thread
            ret, frame = video.read_frame(block=True, timeout=0.1)
            
            if not ret:
                # Stream is dead
                if stream_config.get("camera_id"):
                    print(f"[SYSTEM] Stream {stream_config['camera_id']} died. Exiting for restart.")
                break
            
            if frame is None:
                # Timeout reached or no frame yet; skip this beat
                continue

            clean_frame = frame.copy()
            frame_idx += 1

            # ===== IST TIME & DAILY RESET =====
            now_ist = datetime.now(IST)
            today_str = now_ist.strftime("%Y-%m-%d")
            if last_reset_date != today_str:
                print(f"[SYSTEM] Midnight reset for {today_str} IST.")
                morning_check_done = False
                evening_check_done = False
                auth_success_logged_by_window = {
                    "morning": False,
                    "evening": False,
                }
                active_auth_window = None
                morning_initial_door_checked = False
                evening_auth_started = False
                state_machine.reset_session()
                last_reset_date = today_str

            curr_hour_min = now_ist.strftime("%H:%M")
            
            # Test mode override for auth window
            if test_window:
                is_morning_window = (test_window == "morning")
                is_evening_window = (test_window == "evening")
            else:
                is_morning_window = "06:00" <= curr_hour_min <= "11:00"
                is_evening_window = "12:30" <= curr_hour_min <= "23:58"
            
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
                morning_initial_door_checked = False
                evening_auth_started = False
                tracked_persons = {}
                occupancy_status = "OK"
                auth_result = {
                    "authorized": False,
                    "lock_a_authorized": False,
                    "lock_b_authorized": False,
                    "violation_type": "INCOMPLETE",
                }
                active_auth_window = current_auth_window

            tracking_active = (
                current_auth_window == "morning"
                or (current_auth_window == "evening" and evening_auth_started)
            )

            # ===== PIPELINE =====
            should_process_frame = frame_idx == 1 or (frame_idx - last_processed_frame_idx) >= process_every
            if should_process_frame:
                frame_step = max(frame_idx - last_processed_frame_idx, 1)
                last_processed_frame_idx = frame_idx

                t0 = time.perf_counter()
                if tracking_active:
                    detections = detector.detect(frame)
                    tracked_persons = tracker.update(detections)
                    occupancy_status = state_machine.update_occupancy(tracked_persons, frame_step=frame_step)
                    state_machine.update_timers(tracked_persons, frame_step=frame_step)
                    auth_result = state_machine.check_authorization()
                else:
                    tracked_persons = {}
                    state_machine.active_ids_in_zone = set()
                    state_machine.session["improper_positioning"] = None
                    occupancy_status = "OK"
                    auth_result = {
                        "authorized": False,
                        "lock_a_authorized": False,
                        "lock_b_authorized": False,
                        "violation_type": "INCOMPLETE",
                    }

                is_door_open = False
                ssim_val = None
                if door_verifier:
                    # In morning window, we must ALWAYS check door state to catch the CLOSED->OPEN transition
                    # regardless of whether people are blocking the ROI (grace period handles the check).
                    check_door = (
                        state_machine.should_check_door_state() 
                        or (current_auth_window == "morning" and not morning_check_done)
                        or debug
                    )
                    if check_door:
                        is_door_open = door_verifier.verify(frame)
                    else:
                        is_door_open = last_door_state if last_door_state is not None else False
                        
                    ssim_val = door_verifier.get_last_ssim()
                    door_transition_pending = door_verifier.is_transition_pending()
                else:
                    door_transition_pending = False
                inference_ms = (time.perf_counter() - t0) * 1000.0
                processed_frames_count += 1
                
                # Periodic Telemetry Logging (Production Metrics)
                if frame_idx % 150 == 0:
                    elapsed = time.perf_counter() - t_loop_start
                    actual_fps = processed_frames_count / elapsed if elapsed > 0 else 0
                    telemetry = video.get_telemetry()
                    print(f"[METRICS] {cam_id} | FPS: {actual_fps:.1f} | AI: {inference_ms:.1f}ms | "
                          f"Queue Delay: {telemetry['queue_delay_ms']:.1f}ms | Drops: {telemetry['dropped_frames']}")
                    # Reset counters periodically
                    t_loop_start = time.perf_counter()
                    processed_frames_count = 0

            # ===== VISUALIZATION =====
            draw_rois(visualizer, frame, active_rois)

            unlocker_labels = get_unlocker_labels(state_machine, tracked_persons, frame=frame)
            if debug:
                visible_pose_ids = set(tracked_persons.keys())
            else:
                visible_pose_ids = set(unlocker_labels)

            # Draw only locker-door interaction IDs by default. Raw detections can be
            # enabled for calibration with --show-all-detections or --debug.
            _show_all = show_all_detections or debug
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
            draw_pose_debug(frame, tracked_persons, visible_pose_ids)
            
            # Draw ghost anchors for lost verified persons (ReID search markers)
            draw_lost_verified_ghosts(visualizer, frame, state_machine, unlocker_labels)

            # Draw progress bars at lock centers
            locks_center = roi_manager.get_roi_center("LOCKS_ROI")
            if locks_center:
                id_a_done = state_machine.session.get("id_a") is not None
                id_b_done = state_machine.session.get("id_b") is not None
                
                # Use stream-specific min unlock time for accurate percentage
                pct_a = min((state_machine.session.get("timer_a_seconds", 0) / stream_min_unlock) * 100, 100)
                pct_b = min((state_machine.session.get("timer_b_seconds", 0) / stream_min_unlock) * 100, 100)
                
                if not id_a_done:
                    # Show P1 progress if not yet verified
                    if pct_a > 0:
                        visualizer.draw_circular_progress_bar(frame, tuple(map(int, locks_center)), pct_a)
                elif not id_b_done:
                    # Show P2 progress after P1 is verified
                    if pct_b > 0:
                        visualizer.draw_circular_progress_bar(frame, tuple(map(int, locks_center)), pct_b)
                else:
                    # Both verified, show full circle (green)
                    visualizer.draw_circular_progress_bar(frame, tuple(map(int, locks_center)), 100)

            # Stable Unlockers Count based on assigned sessions
            n = 0
            if tracking_active:
                if state_machine.session.get("candidate_a") is not None or state_machine.session.get("id_a") is not None:
                    n += 1
                if state_machine.session.get("candidate_b") is not None or state_machine.session.get("id_b") is not None:
                    n += 1
            auth_status_text = auth_result["authorized"] if tracking_active else "OFF"
            state_status_text = state_machine.session["sequence_state"] if tracking_active else "IDLE_OUTSIDE_AUDIT"
            visualizer.draw_status_text(
                frame,
                f"Unlockers: {n} | State: {state_status_text} | Auth: {auth_status_text}",
                (10, 30)
            )
            if should_process_frame:
                ai_status = f"AI: {inference_ms:.0f}ms | Every {process_every} frame(s) | IDs only for unlockers"
                if not tracking_active:
                    ai_status = "AI tracking: waiting for OPEN->CLOSE" if current_auth_window == "evening" else "AI tracking: OFF outside audit window"
                visualizer.draw_status_text(
                    frame,
                    ai_status,
                    (10, 55)
                )
            door_status_label = "--" if door_transition_pending else ("OPEN" if is_door_open else "CLOSED")
            if ssim_val is not None:
                visualizer.draw_status_text(frame, f"SSIM: {ssim_val:.3f} | Door: {door_status_label}", (10, 80))

            # Prominent Top-Right Corner Door Status
            door_status_text = f"DOOR: {door_status_label}"
            door_color = (200, 200, 200) if door_transition_pending else ((0, 0, 255) if is_door_open else (0, 255, 0))
            # Use larger font for the corner status
            cv2.putText(frame, door_status_text, (frame.shape[1] - 300, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA) # shadow
            cv2.putText(frame, door_status_text, (frame.shape[1] - 300, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, door_color, 2, cv2.LINE_AA)


            # ===== EVENTS + CAPTURE =====
            site_id = stream_config.get("site_id", "")

            def _capture(event_type, details, check_type="System"):
                capture(
                    alert_system, clean_frame, event_type,
                    evidence_dir=evidence_dir,
                    cam_id=cam_id,
                    site_id=site_id,
                    details=details,
                    check_type=check_type,
                    visualizer=visualizer,
                    unlocker_labels=unlocker_labels,
                    tracked_persons=tracked_persons,
                    auth_result=auth_result,
                    is_door_open=is_door_open,
                    persons_auth_status=persons_auth_status,
                )

            if tracking_active and should_process_frame and occupancy_status == "VIOLATION_OVERCROWD":
                visualizer.draw_status_text(frame, "SECURITY BREACH: Unauthorized Presence",
                                            (10, 80), color=(0, 0, 255), bg_color=(0, 0, 100))
                # Intermediary screenshot disabled to ensure 1 screenshot per window
                # _capture("VIOLATION_OVERCROWD", {"occupancy": len(state_machine.active_ids_in_zone)}, "Security")

            if tracking_active and should_process_frame and auth_result.get("violation_type") == "SAME_ID":
                visualizer.draw_status_text(frame, "SECURITY BREACH: SAME PERSON ATTEMPTING DUAL UNLOCK",
                                            (10, 80), color=(0, 0, 255), bg_color=(0, 0, 100))
                persons_auth_status = False  # Set before capture so screenshot shows UNAVAILABLE
                _capture("DOOR_OPEN_UNAUTHORIZED_PRESENCE", {"reason": "same_person_tried_both_slots"}, current_auth_window or "Security")

                if current_auth_window == "evening":
                    evening_check_done = True
                    evening_auth_started = False
                    print(f"[EVENING] Dual Auth FAILED: Same person attempted both unlocks. Exiting.")
                elif current_auth_window == "morning":
                    # In morning window, if we are transitioning to open, we don't exit immediately.
                    # This allows the post-open grace period to settle and confirm the 2 unlockers.
                    print(f"[MORNING] Potential SAME_ID violation detected (ID {state_machine.session.get('id_a')}). Waiting for door transition.")
                    # Only mark as FAILED if the door is already OPEN and we are not in grace period
                    if is_door_open and not morning_post_open_started:
                        morning_check_done = True
                        print(f"[MORNING] Dual Auth FAILED: Same person attempted both unlocks (Door open). Exiting.")
                
                state_machine.session["violation_type"] = None

            if tracking_active and should_process_frame and state_machine.session.get("improper_positioning"):
                bad_id = state_machine.session["improper_positioning"]
                bad_label = unlocker_labels.get(bad_id, "ignored detection")
                visualizer.draw_status_text(frame, f"IMPROPER POSITIONING: {bad_label}",
                                            (10, 105), color=(0, 165, 255), bg_color=(0, 50, 100))
                # Intermediary screenshot disabled to ensure 1 screenshot per window
                # _capture("IMPROPER_POSITIONING", {"person": bad_label}, "Security")

            # Detect Door Transition
            door_transition = None
            if last_door_state is not None and last_door_state != is_door_open:
                if is_door_open:
                    door_transition = "CLOSED_TO_OPEN"
                else:
                    door_transition = "OPEN_TO_CLOSED"
            last_door_state = is_door_open

            # ===== MORNING CHECK (CLOSED -> OPEN) =====
            # ===== MORNING CHECK (CLOSED -> OPEN) =====
            if is_morning_window and not morning_check_done:
                # 1. Initial State Check
                if not morning_initial_door_checked and not door_transition_pending:
                    morning_initial_door_checked = True
                    if is_door_open:
                        persons_auth_status = False
                        _capture("DOOR_OPENED_EARLIER_THIS_SESSION", {
                            "authorized": False,
                            "door_state": "OPEN",
                            "reason": "door_opened_earlier_this_session",
                        }, "Morning")
                        state_machine.session["door_open_captured"] = True
                        morning_check_done = True
                        print(f"[MORNING] Door already open at {curr_hour_min} IST. Flagging false authentication.")
                        return # Skip further logic for this frame

                # 2. Transition Detection (Checked every frame)
                if door_transition == "CLOSED_TO_OPEN" and not morning_post_open_started:
                    morning_post_open_started = True
                    morning_post_open_start_frame = frame_idx
                    print(f"[MORNING] CLOSED->OPEN detected at {curr_hour_min} IST. "
                          f"Starting {stream_morning_post_open_auth:.0f}s post-open auth window.")
                    
                    # Strict physical presence check: both verified unlockers MUST be in the interaction zone.
                    # prior_auth bypass removed to ensure they are physically confirmed after door opens.
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
                            "timing": "immediate",
                        }, "Morning")
                        print(f"[MORNING] Authorized CLOSED->OPEN confirmed immediately at {curr_hour_min} IST.")
                        state_machine.session["door_open_captured"] = True
                        morning_post_open_started = False
                        morning_check_done = True
                        return

                # 3. Ongoing Grace Window Logic
                if morning_post_open_started:
                    elapsed = (frame_idx - morning_post_open_start_frame) / fps
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
                        state_machine.session["door_open_captured"] = True
                        morning_post_open_started = False
                        morning_check_done = True
                    elif elapsed >= stream_morning_post_open_auth:
                        persons_auth_status = False
                        _capture("DOOR_OPEN_UNAUTHORIZED_PRESENCE", {
                            "authorized": False,
                            "p1_id": state_machine.session.get("id_a"),
                            "p2_id": state_machine.session.get("id_b"),
                            "transition": "CLOSED_TO_OPEN",
                            "both_in_interaction_zone": both_in_interaction,
                            "reason": "missing_dual_auth_or_interaction_zone",
                        }, "Morning")
                        print(f"[MORNING] UNAUTHORIZED CLOSED->OPEN (timeout {elapsed:.1f}s) at {curr_hour_min} IST.")
                        state_machine.session["door_open_captured"] = True
                        morning_post_open_started = False
                        morning_check_done = True
                    else:
                        rem = stream_morning_post_open_auth - elapsed
                        visualizer.draw_status_text(frame, f"MORNING CHECK: CONFIRMING UNLOCKERS ({rem:.1f}s)",
                                                    (10, 130), color=(0, 255, 100), bg_color=(0, 50, 20))
                
                # 4. Idle/Waiting Display (if not in grace window and not done)
                elif not morning_check_done:
                    if auth_result["authorized"]:
                        visualizer.draw_status_text(frame, "MORNING CHECK: 2 UNLOCKERS READY - WAITING FOR CLOSED->OPEN",
                                                    (10, 130), color=(0, 255, 255), bg_color=(0, 50, 50))
                    else:
                        visualizer.draw_status_text(frame, "MORNING CHECK: IDENTIFYING 2 UNLOCKERS",
                                                    (10, 130), color=(0, 165, 255))
            # ===== EVENING CHECK (OPEN -> CLOSED) =====
            elif is_evening_window and not evening_check_done:
                if door_transition == "OPEN_TO_CLOSED" and not evening_auth_started:
                    state_machine.reset_session()
                    evening_auth_started = True
                    tracking_active = True
                    state_machine.session["door_closing_start_frame"] = frame_idx
                    print(f"[EVENING] Door OPEN->CLOSE detected at {curr_hour_min} IST. Starting 5-minute unlocker check.")

                if evening_auth_started:
                    if "door_closing_start_frame" not in state_machine.session or state_machine.session["door_closing_start_frame"] is None:
                        state_machine.session["door_closing_start_frame"] = frame_idx

                    elapsed_frames = frame_idx - state_machine.session["door_closing_start_frame"]
                    elapsed_seconds = elapsed_frames / fps
                    is_auth = auth_result["authorized"]

                    if is_auth:
                        persons_auth_status = True
                        _capture("DOOR_CLOSE_AUTHORIZED_PRESENCE", {
                            "authorized": True,
                            "p1_id": state_machine.session.get("id_a"),
                            "p2_id": state_machine.session.get("id_b"),
                            "wait_time": f"{elapsed_seconds:.1f}s",
                        }, "Evening")
                        print(f"[EVENING] Authorized closure confirmed at {curr_hour_min} IST.")
                        evening_check_done = True
                        evening_auth_started = False
                    elif elapsed_seconds >= stream_evening_second_unlocker_timeout:
                        persons_auth_status = False
                        _capture("DOOR_CLOSE_UNAUTHORIZED_PRESENCE", {
                            "authorized": False,
                            "p1_id": state_machine.session.get("id_a"),
                            "p2_id": state_machine.session.get("id_b"),
                            "wait_time": f"{elapsed_seconds:.1f}s Timeout",
                            "reason": "second_unlocker_timeout",
                        }, "Evening")
                        print(f"[EVENING] UNAUTHORIZED closure (timeout) at {curr_hour_min} IST.")
                        evening_check_done = True
                        evening_auth_started = False
                    else:
                        wait_time_rem = stream_evening_second_unlocker_timeout - elapsed_seconds
                        visualizer.draw_status_text(frame, f"EVENING CHECK: WAITING FOR 2 UNLOCKERS ({wait_time_rem:.0f}s)",
                                                    (10, 130), color=(0, 255, 255), bg_color=(0, 50, 50))
                else:
                    if not is_door_open and not door_transition_pending:
                        evening_check_done = True
                        print(f"[EVENING] Door already closed at {curr_hour_min} IST. Skipping evening check for today.")
                    else:
                        visualizer.draw_status_text(frame, "STATUS: EVENING WINDOW OPEN - WATCHING FOR OPEN->CLOSE",
                                                    (10, 130), color=(0, 165, 255))
            
            else:
                # Idle state
                status_msg = "STATUS: SYSTEM IDLE (OUTSIDE WINDOWS)"
                if not is_morning_window and not is_evening_window:
                    status_msg = f"STATUS: IDLE | NEXT WINDOW: {'MORNING' if curr_hour_min < '06:00' else 'EVENING'}"
                elif morning_check_done and is_morning_window:
                    status_msg = "STATUS: MORNING CHECK COMPLETE"
                elif evening_check_done and is_evening_window:
                    status_msg = "STATUS: EVENING CHECK COMPLETE"

                visualizer.draw_status_text(frame, status_msg, (10, 130), color=(200, 200, 200))

            # Reset opening state machine flag if door closes normally outside of windows
            if not is_door_open:
                state_machine.session["door_open_captured"] = False
                state_machine.session["door_opening_start_frame"] = None
            else:
                # Reset closing flag if door opens
                state_machine.session["door_closing_start_frame"] = None

            if (
                tracking_active
                and should_process_frame
                and auth_result["authorized"]
                and not auth_success_logged_by_window.get(active_auth_window, False)
            ):
                # Window-scoped auth success logging (non-screenshot).
                alert_system.log_event("DUAL_AUTH_SUCCESS", {
                    "window": active_auth_window,
                    "p1_id": state_machine.session.get("id_a"),
                    "p2_id": state_machine.session.get("id_b")
                })
                auth_success_logged_by_window[active_auth_window] = True
                print(f"[SYSTEM] Dual person authorization confirmed for {active_auth_window} window.")

            # ===== TEST WINDOW EXIT =====
            if test_window:
                check_done = (test_window == "morning" and morning_check_done) or \
                             (test_window == "evening" and evening_check_done)
                if check_done:
                    print(f"[SYSTEM] Test window '{test_window}' check complete. Exiting.")
                    break

            # ===== PROGRESS LOG =====
            if (tracking_active or debug) and frame_idx % 30 == 0:
                timers = (f"P1:{state_machine.session['timer_a_seconds']:.1f}s "
                          f"P2:{state_machine.session['timer_b_seconds']:.1f}s")
                cand_a = f"ID {state_machine.session['candidate_a']}" if state_machine.session["candidate_a"] is not None else "-"
                cand_b = f"ID {state_machine.session['candidate_b']}" if state_machine.session["candidate_b"] is not None else "-"
                id_a = f"ID {state_machine.session['id_a']}" if state_machine.session["id_a"] is not None else "-"
                id_b = f"ID {state_machine.session['id_b']}" if state_machine.session["id_b"] is not None else "-"
                ssim_str = f" | Door SSIM: {ssim_val:.3f}" if ssim_val is not None else ""
                print(f"[PROGRESS] Frame {frame_idx}/{total_frames} ({video.get_progress():.1f}%) "
                      f"| Unlockers: {n} | State: {state_machine.session['sequence_state']} "
                      f"| Candidates: P1={cand_a} P2={cand_b} "
                      f"| Verified: P1={id_a} P2={id_b} | {timers}{ssim_str}")

            if live_window_available:
                try:
                    if debug:
                        display_frame = frame
                    else:
                        display_frame = clean_frame.copy()
                        visualizer.draw_client_overlays(
                            display_frame, unlocker_labels, tracked_persons, auth_result, is_door_open,
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
    parser.add_argument("--stream-index", type=int, default=None, help="Index of the stream config to use from config.STREAMS_CONFIG. If omitted, runs all streams in parallel.")
    parser.add_argument("video_source", nargs="?", default=None, help="Video file path, RTSP stream, or webcam index (overrides config).")
    parser.add_argument("--show", action="store_true", help="Enable live OpenCV preview window.")
    parser.add_argument(
        "--scale-rois",
        action="store_true",
        help="Scale the RTSP-calibrated 2688x1520 ROIs to a different video resolution.",
    )
    parser.add_argument(
        "--process-every",
        type=int,
        default=3,
        help="Run pose inference every N frames for smoother live preview. Use 1 for full per-frame analysis.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device. Use cuda to force the system GPU.",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable CUDA half precision inference.",
    )
    parser.add_argument(
        "--show-all-detections",
        action="store_true",
        help="Show unlabeled raw person detections for calibration/debugging.",
    )
    parser.add_argument(
        "--test-window",
        type=str,
        choices=["morning", "evening"],
        default=None,
        help="Test mode: force morning or evening window logic regardless of current time.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show all debug overlays on live window (ROIs, SSIM, AI stats, all detections). Screenshots remain clean.",
    )
    args = parser.parse_args()

    if args.stream_index is None:
        if args.video_source is not None:
            # Backwards compatibility: if testing a specific video without stream-index, default to 0
            args.stream_index = 0
        else:
            # No parameters provided -> run all streams in parallel
            import subprocess
            print(f"[SYSTEM] No stream parameters provided. Launching all {len(config.STREAMS_CONFIG)} streams in parallel...")
            processes = []
            
            base_cmd = [sys.executable, sys.argv[0]]
            if args.show: 
                print("[WARNING] --show enabled. OpenCV GUI rendering adds significant CPU/latency overhead and is NOT recommended for production.")
                base_cmd.append("--show")
            if args.scale_rois: base_cmd.append("--scale-rois")
            base_cmd.extend(["--process-every", str(args.process_every)])
            base_cmd.extend(["--device", args.device])
            if args.no_half: base_cmd.append("--no-half")
            if args.show_all_detections: base_cmd.append("--show-all-detections")
            if args.test_window: base_cmd.extend(["--test-window", args.test_window])
            if args.debug: base_cmd.append("--debug")

            for i in range(len(config.STREAMS_CONFIG)):
                cmd = base_cmd + ["--stream-index", str(i)]
                p = subprocess.Popen(cmd)
                processes.append((p, cmd, i))
                print(f"[SYSTEM] Launched Stream {i} (PID: {p.pid})")
                
                # Staggered Startup (Production Safety)
                if i < len(config.STREAMS_CONFIG) - 1:
                    delay = getattr(config, "STAGGER_START_DELAY", 2.0)
                    print(f"[SYSTEM] Waiting {delay}s before next launch...")
                    time.sleep(delay)
                
            print("[SYSTEM] All streams launched. Supervisor active.")
            try:
                while True:
                    time.sleep(5)
                    # Watchdog / Supervisor logic
                    for idx, (p, cmd, s_idx) in enumerate(processes):
                        if p.poll() is not None:
                            print(f"[WATCHDOG] Stream {s_idx} (PID: {p.pid}) died with code {p.returncode}. Restarting...")
                            new_p = subprocess.Popen(cmd)
                            processes[idx] = (new_p, cmd, s_idx)
                            print(f"[WATCHDOG] Stream {s_idx} restarted (New PID: {new_p.pid})")
            except KeyboardInterrupt:
                print("\n[SYSTEM] Shutting down all streams...")
                for p, _, _ in processes:
                    p.terminate()
                    try:
                        p.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        print(f"[SYSTEM] Force killing PID {p.pid}")
                        p.kill()
            sys.exit(0)

    if args.stream_index < 0 or args.stream_index >= len(config.STREAMS_CONFIG):
        print(f"[ERROR] Invalid stream-index {args.stream_index}. Available streams: 0 to {len(config.STREAMS_CONFIG)-1}.")
        sys.exit(1)

    stream_config = config.STREAMS_CONFIG[args.stream_index]

    video_source = args.video_source
    if video_source is not None and video_source.isdigit():
        video_source = int(video_source)

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

