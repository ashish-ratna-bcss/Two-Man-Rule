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

CAM_ID = config.RTSP_URLS[0]["camera_id"]
SITE_NAME = config.RTSP_URLS[0]["site_name"]
EVIDENCE_DIR = os.path.join(config.BASE_OUTPUT_DIR, SITE_NAME, CAM_ID)
LOG_DIR = "logs"
CALIBRATED_W, CALIBRATED_H = 2688, 1520

_alert_counter = [0]  # mutable so capture() can increment across calls


def _scale_polygon(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale RTSP-calibrated polygon coordinates to another source frame size."""
    scale = np.array([width / CALIBRATED_W, height / CALIBRATED_H], dtype=np.float32)
    return np.rint(points.astype(np.float32) * scale).astype(np.int32)


def setup_rois(roi_manager: ROIManager, width: int, height: int, scale_rois: bool = False):
    """Register ROIs in the same coordinate space as the original video frame."""
    transform = (lambda points: _scale_polygon(points, width, height)) if scale_rois else (
        lambda points: points.astype(np.int32).copy()
    )
    rois = {
        "LOCK_A_ROI": transform(config.LOCK_A_ROI),
        "LOCK_B_ROI": transform(config.LOCK_B_ROI),
        "DOOR_ROI": transform(config.DOOR_ROI),
        "STANDING_ZONE": transform(config.STANDING_ZONE),
        "INTERACTION_ZONE": transform(config.INTERACTION_ZONE),
        "DOOR_CORNER_ROI": transform(config.DOOR_CORNER_ROI),
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
        "LOCK_A_ROI": ((0, 255, 0), 2),
        "LOCK_B_ROI": ((0, 255, 0), 2),
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

    # 3. Anchor + size fallback
    anchor = state_machine.verified_anchors.get(slot)
    ref_bbox = state_machine.last_seen_bbox.get(slot)
    if anchor is None:
        return

    wide_threshold = config.UNLOCKER_ANCHOR_MATCH_PIXELS * 5
    best_dist = float("inf")
    best_tid = None
    best_bbox = None

    for tid, person in candidates.items():
        bbox = person.get("bbox")
        if bbox is None:
            continue
        cx = (bbox[0] + bbox[2]) / 2
        cy = float(bbox[3])
        dist = math.dist((cx, cy), anchor)
        if dist < best_dist:
            best_dist = dist
            best_tid = tid
            best_bbox = bbox

    if best_tid is not None and best_dist <= wide_threshold:
        if not _bbox_size_matches(ref_bbox, best_bbox, tolerance=0.4):
            return
        labels[best_tid] = f"{slot_label} (remapped ID {primary_id})"
        state_machine.assign_unlocker_tag(best_tid, slot)
        print(f"[VIZ] {slot_label} anchor-remapped → ID {best_tid} (dist={best_dist:.1f})")


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
    details: dict = None,
    check_type: str = "System",
    visualizer: "Visualizer" = None,
    unlocker_labels: dict = None,
    tracked_persons: dict = None,
    auth_result: dict = None,
    is_door_open: bool = False,
    persons_auth_status=None,
):
    """Unified capture + log helper. Saves clean client frame with 2 status panels."""
    now_ist = datetime.now(IST)
    date_str = now_ist.strftime("%d-%m-%Y")
    time_str = now_ist.strftime("%H-%M-%S")

    target_dir = os.path.join(EVIDENCE_DIR, date_str)
    os.makedirs(target_dir, exist_ok=True)

    _alert_counter[0] += 1
    filename = f"alert_{_alert_counter[0]}_{CAM_ID}_{date_str}_{time_str}.png"
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
    alert_system.log_event(event_type, details or {})
    print(f"[CAPTURE] {event_type}: {full_path} (write={'OK' if ok else 'FAILED'})")
    return full_path


def main(
    video_source: str,
    show_live: bool = True,
    scale_rois: bool = False,
    process_every: int = 3,
    device: str = "auto",
    half: bool = True,
    show_all_detections: bool = False,
    test_window: str = None,
    debug: bool = False,
):
    print("[SYSTEM] Initializing Two-Man Rule Monitoring System...")
    _alert_counter[0] = 0
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

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

        active_rois = setup_rois(roi_manager, width, height, scale_rois=scale_rois)
        print_roi_coordinates(active_rois, width, height, scale_rois)
        try:
            door_verifier = DoorVerifier(
                config.CLOSED_DOOR_REFERENCE,
                similarity_threshold=config.SSIM_THRESHOLD
            )
            print(f"[SYSTEM] Door verifier loaded with threshold {config.SSIM_THRESHOLD}")
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")
            door_verifier = None

        state_machine = DualAuthStateMachine(roi_manager, int(fps))
        visualizer = Visualizer()
        alert_system = AlertSystem(evidence_dir=EVIDENCE_DIR, log_dir=LOG_DIR)

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
            # If started after 10:30 AM, morning check is neglected for today
            morning_check_done = curr_hm > "10:30"
            # If started after 11:00 PM, evening check is neglected for today
            evening_check_done = curr_hm > "23:00"
            
            if morning_check_done:
                print(f"[SYSTEM] Startup after 10:30 AM IST. Morning check for {last_reset_date} marked as SKIPPED.")
            if evening_check_done:
                print(f"[SYSTEM] Startup after 11:00 PM IST. Evening check for {last_reset_date} marked as SKIPPED.")
        
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
            cv2.namedWindow("Two-Man Rule Live ROI Debug", cv2.WINDOW_NORMAL)

        print("[SYSTEM] Starting frame processing loop...")

        while True:
            ret, frame = video.read_frame()
            if not ret:
                break

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
                is_morning_window = "09:30" <= curr_hour_min <= "10:30"
                is_evening_window = "20:30" <= curr_hour_min <= "23:00"
            
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
                    is_door_open = door_verifier.verify(frame)
                    ssim_val = door_verifier.get_last_ssim()
                    door_transition_pending = door_verifier.is_transition_pending()
                else:
                    door_transition_pending = False
                inference_ms = (time.perf_counter() - t0) * 1000.0

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

            # Draw progress bars at lock centers
            lock_a_center = roi_manager.get_roi_center("LOCK_A_ROI")
            lock_b_center = roi_manager.get_roi_center("LOCK_B_ROI")
            if lock_a_center:
                pct = min((state_machine.session["timer_a_seconds"] / config.MIN_UNLOCK_SECONDS) * 100, 100)
                visualizer.draw_circular_progress_bar(frame, tuple(map(int, lock_a_center)), pct)
            if lock_b_center:
                pct = min((state_machine.session["timer_b_seconds"] / config.MIN_UNLOCK_SECONDS) * 100, 100)
                visualizer.draw_circular_progress_bar(frame, tuple(map(int, lock_b_center)), pct)

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

            if not roi_preview_saved:
                preview_path = os.path.join(EVIDENCE_DIR, f"ROI_PREVIEW_{CAM_ID}.jpg")
                os.makedirs(EVIDENCE_DIR, exist_ok=True)
                cv2.imwrite(preview_path, frame)
                print(f"[ROI] ROI preview saved: {preview_path}")
                roi_preview_saved = True

            # ===== EVENTS + CAPTURE =====
            def _capture(event_type, details, check_type="System"):
                capture(
                    alert_system, clean_frame, event_type, details,
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
                _capture("VIOLATION_SAME_PERSON", {"reason": "same_person_tried_both_slots"}, "Security")
                
                if current_auth_window == "evening":
                    persons_auth_status = False
                    evening_check_done = True
                    evening_auth_started = False
                    print(f"[EVENING] Dual Auth FAILED: Same person attempted both unlocks. Exiting.")
                elif current_auth_window == "morning":
                    persons_auth_status = False
                    morning_check_done = True
                    morning_post_open_started = False
                    print(f"[MORNING] Dual Auth FAILED: Same person attempted both unlocks. Exiting.")
                
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
            if is_morning_window and not morning_check_done:
                if not morning_initial_door_checked and not door_transition_pending:
                    morning_initial_door_checked = True
                    if is_door_open:
                        persons_auth_status = False  # door already open = no proper 2-person auth
                        _capture("DOOR_OPENED_EARLIER_THIS_SESSION", {
                            "authorized": False,
                            "door_state": "OPEN",
                            "reason": "door_opened_earlier_this_session",
                        }, "Morning")
                        state_machine.session["door_open_captured"] = True
                        morning_check_done = True
                        print(f"[MORNING] Door already open at {curr_hour_min} IST. Flagging false authentication.")
                    else:
                        visualizer.draw_status_text(frame, "MORNING CHECK: IDENTIFYING 2 UNLOCKERS",
                                                    (10, 130), color=(0, 165, 255))
                elif door_transition == "CLOSED_TO_OPEN" and not morning_post_open_started:
                    morning_post_open_started = True
                    morning_post_open_start_frame = frame_idx
                    print(f"[MORNING] CLOSED->OPEN detected at {curr_hour_min} IST. "
                          f"Starting {config.MORNING_POST_OPEN_AUTH_SECONDS:.0f}s post-open auth window.")
                    visualizer.draw_status_text(frame, "MORNING CHECK: DOOR OPENED - CONFIRMING UNLOCKERS",
                                                (10, 130), color=(0, 255, 100), bg_color=(0, 50, 20))
                elif morning_post_open_started:
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
                    elif elapsed >= config.MORNING_POST_OPEN_AUTH_SECONDS:
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
                        rem = config.MORNING_POST_OPEN_AUTH_SECONDS - elapsed
                        visualizer.draw_status_text(frame, f"MORNING CHECK: CONFIRMING UNLOCKERS ({rem:.1f}s)",
                                                    (10, 130), color=(0, 255, 100), bg_color=(0, 50, 20))
                elif auth_result["authorized"]:
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
                    elif elapsed_seconds >= config.EVENING_SECOND_UNLOCKER_TIMEOUT_SECONDS:
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
                        wait_time_rem = config.EVENING_SECOND_UNLOCKER_TIMEOUT_SECONDS - elapsed_seconds
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
                    status_msg = f"STATUS: IDLE | NEXT WINDOW: {'MORNING' if curr_hour_min < '09:30' else 'EVENING'}"
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

            # ===== DEBUG FRAME: save annotated frame when first unlocker is detected =====
            if tracking_active and should_process_frame and not debug_frame_saved and len(state_machine.active_ids_in_zone) >= 1:
                debug_frame = frame.copy()
                # Draw all keypoints for all tracked persons
                for track_id, person in tracked_persons.items():
                    if track_id not in unlocker_labels:
                        continue
                    kpts = person["keypoints"]
                    for kp_idx, (kx, ky, kc) in enumerate(kpts):
                        if kc > 0.3:
                            cv2.circle(debug_frame, (int(kx), int(ky)), 4, (0, 255, 255), -1)
                            cv2.putText(debug_frame, str(kp_idx), (int(kx)+4, int(ky)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1)
                    # Draw bbox centroid
                    bbox = person["bbox"]
                    cx = int((bbox[0]+bbox[2])/2)
                    cy = int((bbox[1]+bbox[3])/2)
                    cv2.circle(debug_frame, (cx, cy), 8, (0, 0, 255), -1)
                    label = unlocker_labels.get(track_id, "ignored")
                    cv2.putText(debug_frame, f"{label} center", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                debug_path = os.path.join(EVIDENCE_DIR, f"DEBUG_ROI_frame{frame_idx}.jpg")
                cv2.imwrite(debug_path, debug_frame)
                print(f"[DEBUG] ROI debug frame saved: {debug_path}")
                print(f"[DEBUG] Scaled ROI centers:")
                for name in ["LOCK_A_ROI", "LOCK_B_ROI", "STANDING_ZONE", "INTERACTION_ZONE"]:
                    c = roi_manager.get_roi_center(name)
                    print(f"  {name}: center={c}")
                print(f"[DEBUG] Tracked persons keypoints (wrist R=10, wrist L=9, ankle R=16, ankle L=15):")
                for track_id, person in tracked_persons.items():
                    if track_id not in unlocker_labels:
                        continue
                    kpts = person["keypoints"]
                    bbox = person["bbox"]
                    cx = (bbox[0]+bbox[2])/2
                    cy = (bbox[1]+bbox[3])/2
                    wr = kpts[config.KEYPOINT_WRIST_RIGHT]
                    wl = kpts[config.KEYPOINT_WRIST_LEFT]
                    ar = kpts[config.KEYPOINT_ANKLE_RIGHT]
                    al = kpts[config.KEYPOINT_ANKLE_LEFT]
                    label = unlocker_labels.get(track_id, "ignored")
                    print(f"  {label}: center=({cx:.0f},{cy:.0f}) | wrist_R=({wr[0]:.0f},{wr[1]:.0f},c={wr[2]:.2f}) | wrist_L=({wl[0]:.0f},{wl[1]:.0f},c={wl[2]:.2f}) | ankle_R=({ar[0]:.0f},{ar[1]:.0f},c={ar[2]:.2f})")
                debug_frame_saved = True

            # ===== TEST WINDOW EXIT =====
            if test_window:
                check_done = (test_window == "morning" and morning_check_done) or \
                             (test_window == "evening" and evening_check_done)
                if check_done:
                    print(f"[SYSTEM] Test window '{test_window}' check complete. Exiting.")
                    break

            # ===== PROGRESS LOG =====
            if tracking_active and frame_idx % 30 == 0:
                timers = (f"P1:{state_machine.session['timer_a_seconds']:.1f}s "
                          f"P2:{state_machine.session['timer_b_seconds']:.1f}s")
                cand_a = f"ID {state_machine.session['candidate_a']}" if state_machine.session["candidate_a"] is not None else "-"
                cand_b = f"ID {state_machine.session['candidate_b']}" if state_machine.session["candidate_b"] is not None else "-"
                id_a = f"ID {state_machine.session['id_a']}" if state_machine.session["id_a"] is not None else "-"
                id_b = f"ID {state_machine.session['id_b']}" if state_machine.session["id_b"] is not None else "-"
                print(f"[PROGRESS] Frame {frame_idx}/{total_frames} ({video.get_progress():.1f}%) "
                      f"| Unlockers: {n} | State: {state_machine.session['sequence_state']} "
                      f"| Candidates: P1={cand_a} P2={cand_b} "
                      f"| Verified: P1={id_a} P2={id_b} | {timers}")

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
                    cv2.imshow("Two-Man Rule Live ROI Debug", display_frame)
                    wait_ms = max(1, int(1000 / max(fps, 1)))
                    if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                        print("[SYSTEM] Live preview stopped by user.")
                        break
                except cv2.error as e:
                    print(f"[WARNING] Live preview unavailable: {e}")
                    live_window_available = False

    print("[SYSTEM] Processing complete.")
    log_path = alert_system.save_session_log()
    print(f"[SYSTEM] Session log: {log_path}")
    print(f"[SYSTEM] Evidence files: {len(os.listdir(EVIDENCE_DIR))}")
    if 'live_window_available' in locals() and live_window_available:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Two-Man Rule monitoring with live ROI overlay.")
    default_url = config.RTSP_URLS[0]["rtsp_url"]
    parser.add_argument("video_source", nargs="?", default=default_url, help="Video file path, RTSP stream, or webcam index.")
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

    video_source = int(args.video_source) if str(args.video_source).isdigit() else args.video_source
    main(
        video_source,
        show_live=args.show,
        scale_rois=args.scale_rois,
        process_every=args.process_every,
        device=args.device,
        half=not args.no_half,
        show_all_detections=args.show_all_detections,
        test_window=args.test_window,
        debug=args.debug,
    )
