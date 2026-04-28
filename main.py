# main.py
import sys
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

EVIDENCE_DIR = "logs/evidence"
LOG_DIR = "logs"
CALIBRATED_W, CALIBRATED_H = 2688, 1520


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


def capture(alert_system: AlertSystem, frame: np.ndarray, event_type: str, details: dict = None):
    """Unified capture + log helper."""
    filepath = alert_system.evidence_dir
    ts = alert_system._filename_timestamp()
    filename = f"{event_type}_{ts}.jpg"
    full_path = os.path.join(filepath, filename)
    ok = cv2.imwrite(full_path, frame)
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
):
    print("[SYSTEM] Initializing Two-Man Rule Monitoring System...")
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
            door_verifier = DoorVerifier(config.CLOSED_DOOR_REFERENCE)
            print("[SYSTEM] Door verifier loaded.")
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")
            door_verifier = None

        state_machine = DualAuthStateMachine(roi_manager, int(fps))
        visualizer = Visualizer()
        alert_system = AlertSystem(evidence_dir=EVIDENCE_DIR, log_dir=LOG_DIR)

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
        is_door_open = False
        ssim_val = None
        live_window_available = can_show_live_window(show_live)
        if live_window_available:
            cv2.namedWindow("Two-Man Rule Live ROI Debug", cv2.WINDOW_NORMAL)

        print("[SYSTEM] Starting frame processing loop...")

        while True:
            ret, frame = video.read_frame()
            if not ret:
                break

            frame_idx += 1

            # ===== PIPELINE =====
            should_process_frame = frame_idx == 1 or (frame_idx - last_processed_frame_idx) >= process_every
            if should_process_frame:
                frame_step = max(frame_idx - last_processed_frame_idx, 1)
                last_processed_frame_idx = frame_idx

                t0 = time.perf_counter()
                detections = detector.detect(frame)
                tracked_persons = tracker.update(detections)
                occupancy_status = state_machine.update_occupancy(tracked_persons, frame_step=frame_step)
                state_machine.update_timers(tracked_persons, frame_step=frame_step)
                auth_result = state_machine.check_authorization()

                is_door_open = False
                ssim_val = None
                if door_verifier:
                    is_door_open = door_verifier.verify(frame)
                    ssim_val = door_verifier.get_last_ssim()
                inference_ms = (time.perf_counter() - t0) * 1000.0

            # ===== VISUALIZATION =====
            draw_rois(visualizer, frame, active_rois)

            unlocker_labels = get_unlocker_labels(state_machine, tracked_persons, frame=frame)
            visible_pose_ids = set(unlocker_labels)

            # Draw only locker-door interaction IDs by default. Raw detections can be
            # enabled for calibration with --show-all-detections.
            for track_id, person in tracked_persons.items():
                if track_id in unlocker_labels:
                    label = unlocker_labels[track_id]
                elif show_all_detections:
                    label = ""
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
            if state_machine.session.get("candidate_a") is not None or state_machine.session.get("id_a") is not None:
                n += 1
            if state_machine.session.get("candidate_b") is not None or state_machine.session.get("id_b") is not None:
                n += 1
            visualizer.draw_status_text(
                frame,
                f"Unlockers: {n} | State: {state_machine.session['sequence_state']} | Auth: {auth_result['authorized']}",
                (10, 30)
            )
            if should_process_frame:
                visualizer.draw_status_text(
                    frame,
                    f"AI: {inference_ms:.0f}ms | Every {process_every} frame(s) | IDs only for unlockers",
                    (10, 55)
                )
            if ssim_val is not None:
                visualizer.draw_status_text(frame, f"SSIM: {ssim_val:.3f} | Door: {'OPEN' if is_door_open else 'CLOSED'}", (10, 80))

            # Prominent Top-Right Corner Door Status
            door_status_text = f"DOOR: {'OPEN' if is_door_open else 'CLOSED'}"
            door_color = (0, 0, 255) if is_door_open else (0, 255, 0) # Red if open, Green if closed
            # Use larger font for the corner status
            cv2.putText(frame, door_status_text, (frame.shape[1] - 300, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA) # shadow
            cv2.putText(frame, door_status_text, (frame.shape[1] - 300, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, door_color, 2, cv2.LINE_AA)

            if not roi_preview_saved:
                preview_path = os.path.join(EVIDENCE_DIR, "ROI_PREVIEW_first_frame.jpg")
                cv2.imwrite(preview_path, frame)
                print(f"[ROI] First annotated ROI preview saved: {preview_path}")
                roi_preview_saved = True

            # ===== EVENTS + CAPTURE =====
            if should_process_frame and occupancy_status == "VIOLATION_OVERCROWD":
                visualizer.draw_status_text(frame, "SECURITY BREACH: Unauthorized Presence",
                                            (10, 80), color=(0, 0, 255), bg_color=(0, 0, 100))
                capture(alert_system, frame, "VIOLATION_OVERCROWD",
                        {"occupancy": len(state_machine.active_ids_in_zone)})

            if should_process_frame and state_machine.session.get("improper_positioning"):
                bad_id = state_machine.session["improper_positioning"]
                bad_label = unlocker_labels.get(bad_id, "ignored detection")
                visualizer.draw_status_text(frame, f"IMPROPER POSITIONING: {bad_label}",
                                            (10, 105), color=(0, 165, 255), bg_color=(0, 50, 100))
                capture(alert_system, frame, "IMPROPER_POSITIONING", {"person": bad_label})

            if should_process_frame and is_door_open:
                if not state_machine.session.get("door_open_captured"):
                    if "door_opening_start_frame" not in state_machine.session or state_machine.session["door_opening_start_frame"] is None:
                        state_machine.session["door_opening_start_frame"] = frame_idx
                    
                    # Calculate elapsed time
                    elapsed_frames = frame_idx - state_machine.session["door_opening_start_frame"]
                    elapsed_seconds = elapsed_frames / fps
                    
                    is_auth = auth_result["authorized"]
                    
                    id_a = state_machine.session.get("id_a")
                    id_b = state_machine.session.get("id_b")
                    # Use stable assigned IDs, avoiding flickering keypoint checks
                    unlockers_present = sum(1 for tid in [id_a, id_b] if tid is not None)
                    
                    # Condition 1: Auth becomes True ANY TIME within the 5 seconds
                    if is_auth:
                        capture(alert_system, frame, "DOOR_OPEN_AUTHORIZED_PRESENCE", {
                            "authorized": True,
                            "p1_id": id_a,
                            "p2_id": id_b,
                            "unlockers_present": unlockers_present,
                            "wait_time": f"{elapsed_seconds:.1f}s"
                        })
                        state_machine.session["door_open_captured"] = True
                        print(f"\n[SYSTEM] Authorized within 5s grace period ({elapsed_seconds:.1f}s). Evidence saved. Exiting.")
                        import sys
                        sys.exit(0)
                        
                    # Condition 2: 5 seconds have elapsed and Auth is still False
                    elif elapsed_seconds >= 5.0:
                        capture(alert_system, frame, "DOOR_OPEN_UNAUTHORIZED_PRESENCE", {
                            "authorized": False,
                            "p1_id": id_a,
                            "p2_id": id_b,
                            "unlockers_present": unlockers_present,
                            "wait_time": "5.0s Timeout"
                        })
                        state_machine.session["door_open_captured"] = True
                        print(f"\n[SYSTEM] 5s grace period expired without authorization (Unlockers: {unlockers_present}). UNAUTHORIZED evidence saved. Exiting.")
                        import sys
                        sys.exit(0)
                        
                    # Condition 3: Still waiting in the 5 second grace period
                    else:
                        wait_time_rem = 5.0 - elapsed_seconds
                        visualizer.draw_status_text(frame, f"DOOR OPEN: WAITING FOR AUTH ({wait_time_rem:.1f}s)",
                                                    (10, 130), color=(0, 255, 255), bg_color=(0, 50, 50))
            elif should_process_frame and not is_door_open:
                state_machine.session["door_open_captured"] = False
                state_machine.session["door_opening_start_frame"] = None

            if should_process_frame and auth_result["authorized"] and not state_machine.session.get("auth_success_logged"):
                capture(alert_system, frame, "DUAL_AUTH_SUCCESS", {
                    "person_a": "P1",
                    "person_b": "P2",
                    "timer_a_s": state_machine.session["timer_a_seconds"],
                    "timer_b_s": state_machine.session["timer_b_seconds"],
                    "sequence_state": state_machine.session["sequence_state"],
                })
                state_machine.session["auth_success_logged"] = True

            # ===== DEBUG FRAME: save annotated frame when first unlocker is detected =====
            if should_process_frame and not debug_frame_saved and len(state_machine.active_ids_in_zone) >= 1:
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

            # ===== PROGRESS LOG =====
            if frame_idx % 30 == 0:
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
                    cv2.imshow("Two-Man Rule Live ROI Debug", frame)
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
    parser.add_argument("video_source", nargs="?", default=0, help="Video file path or webcam index.")
    parser.add_argument("--no-show", action="store_true", help="Disable live OpenCV preview window.")
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
    args = parser.parse_args()

    video_source = int(args.video_source) if str(args.video_source).isdigit() else args.video_source
    main(
        video_source,
        show_live=not args.no_show,
        scale_rois=args.scale_rois,
        process_every=args.process_every,
        device=args.device,
        half=not args.no_half,
        show_all_detections=args.show_all_detections,
    )
