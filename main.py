# main.py
import sys
import cv2
import numpy as np
from models.pose_detector import PoseDetector
from models.tracker import PersonTracker
from models.door_verifier import DoorVerifier
from logic.roi_manager import ROIManager
from logic.state_machine import DualAuthStateMachine
from io.video_handler import VideoHandler
from io.visualizer import Visualizer
from io.alert_system import AlertSystem
import config

def setup_rois(roi_manager: ROIManager):
    """Register ROIs from config."""
    if config.LOCK_A_ROI:
        roi_manager.register_rect_roi(
            "LOCK_A_ROI",
            config.LOCK_A_ROI[0], config.LOCK_A_ROI[1],
            config.LOCK_A_ROI[2], config.LOCK_A_ROI[3]
        )

    if config.LOCK_B_ROI:
        roi_manager.register_rect_roi(
            "LOCK_B_ROI",
            config.LOCK_B_ROI[0], config.LOCK_B_ROI[1],
            config.LOCK_B_ROI[2], config.LOCK_B_ROI[3]
        )

    if config.DOOR_ROI:
        roi_manager.register_polygon_roi("DOOR_ROI", config.DOOR_ROI)

    if config.INTERACTION_ZONE:
        roi_manager.register_polygon_roi("INTERACTION_ZONE", config.INTERACTION_ZONE)

def main(video_source: str):
    """Main processing pipeline."""
    print("[SYSTEM] Initializing Two-Man Rule Monitoring System...")

    # Check ROI configuration
    if not all([config.LOCK_A_ROI, config.LOCK_B_ROI, config.DOOR_ROI, config.INTERACTION_ZONE]):
        print("[ERROR] ROI configuration incomplete. Please provide:")
        print(f"  - LOCK_A_ROI: {config.LOCK_A_ROI}")
        print(f"  - LOCK_B_ROI: {config.LOCK_B_ROI}")
        print(f"  - DOOR_ROI: {config.DOOR_ROI}")
        print(f"  - INTERACTION_ZONE: {config.INTERACTION_ZONE}")
        sys.exit(1)

    # Initialize components
    print("[SYSTEM] Loading models...")
    detector = PoseDetector()
    tracker = PersonTracker()

    try:
        door_verifier = DoorVerifier(config.CLOSED_DOOR_REFERENCE)
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
        door_verifier = None

    roi_manager = ROIManager()
    setup_rois(roi_manager)

    with VideoHandler(video_source) as video:
        fps = video.get_fps()
        width, height = video.get_dimensions()
        total_frames = video.get_total_frames()

        print(f"[VIDEO] FPS: {fps}, Resolution: {width}x{height}, Total Frames: {total_frames}")

        state_machine = DualAuthStateMachine(roi_manager, int(fps))
        visualizer = Visualizer()
        alert_system = AlertSystem()

        frame_idx = 0
        door_state_last = None

        print("[SYSTEM] Starting frame processing loop...")

        while True:
            ret, frame = video.read_frame()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % 30 == 0:  # Log every 30 frames
                print(f"[PROGRESS] Frame {frame_idx}/{total_frames} ({video.get_progress():.1f}%)")

            # ===== PIPELINE =====

            # 1. Pose Detection
            detections = detector.detect(frame)

            # 2. Tracking
            tracked_persons = tracker.update(detections)

            # 3. Occupancy Update (Census)
            occupancy_status = state_machine.update_occupancy(tracked_persons)

            # 4. Timer Update (only if occupancy == 2)
            state_machine.update_timers(tracked_persons)

            # 5. Check Authorization
            auth_result = state_machine.check_authorization()

            # 6. Door Verification (if applicable)
            if door_verifier and len(state_machine.active_ids_in_zone) <= 1:
                is_door_open = door_verifier.is_door_open(frame)
                ssim = door_verifier.get_last_ssim()
            else:
                is_door_open = False
                ssim = None

            # ===== VISUALIZATION =====

            # Draw tracked persons
            for track_id, person in tracked_persons.items():
                bbox = person["bbox"]

                # Color code by state
                if track_id in state_machine.active_ids_in_zone:
                    if auth_result["authorized"]:
                        color = config.COLOR_AUTHORIZED
                    elif auth_result["lock_a_authorized"] or auth_result["lock_b_authorized"]:
                        color = config.COLOR_UNLOCKING
                    else:
                        color = config.COLOR_DETECTED
                else:
                    color = config.COLOR_DETECTED

                label = f"ID {track_id}"
                visualizer.draw_bounding_box(frame, bbox, color, label)

            # Draw progress bars at lock ROIs
            lock_a_center = roi_manager.get_roi_center("LOCK_A_ROI")
            lock_b_center = roi_manager.get_roi_center("LOCK_B_ROI")

            if lock_a_center:
                progress_a = (state_machine.session["timer_a_seconds"] / 10.0) * 100
                visualizer.draw_circular_progress_bar(
                    frame,
                    tuple(map(int, lock_a_center)),
                    progress_a,
                    color=config.COLOR_UNLOCKING if progress_a > 0 else config.COLOR_DETECTED
                )

            if lock_b_center:
                progress_b = (state_machine.session["timer_b_seconds"] / 10.0) * 100
                visualizer.draw_circular_progress_bar(
                    frame,
                    tuple(map(int, lock_b_center)),
                    progress_b,
                    color=config.COLOR_UNLOCKING if progress_b > 0 else config.COLOR_DETECTED
                )

            # Draw ROI outlines
            if config.INTERACTION_ZONE:
                points = np.array(config.INTERACTION_ZONE, dtype=np.int32)
                visualizer.draw_roi_polygon(frame, points, (100, 100, 255), 1)

            # Draw status
            status_text = f"Occupancy: {len(state_machine.active_ids_in_zone)} | Auth: {auth_result['authorized']}"
            visualizer.draw_status_text(frame, status_text, (10, 30), color=(255, 255, 255))

            if ssim is not None:
                ssim_text = f"SSIM: {ssim:.3f} | Door: {'OPEN' if is_door_open else 'CLOSED'}"
                visualizer.draw_status_text(frame, ssim_text, (10, 60), color=(255, 255, 255))

            if occupancy_status == "VIOLATION_OVERCROWD":
                visualizer.draw_status_text(
                    frame,
                    "SECURITY BREACH: Unauthorized Presence",
                    (10, 90),
                    color=(0, 0, 255),
                    bg_color=(0, 0, 100)
                )
                alert_system.capture_violation_overcrowd(frame)
                alert_system.log_event("OVERCROWD_DETECTED")

            # Check authorization and log
            if auth_result["authorized"]:
                alert_system.log_event("DUAL_AUTH_COMPLETE", {
                    "id_a": state_machine.session["id_a"],
                    "id_b": state_machine.session["id_b"],
                    "timer_a_seconds": state_machine.session["timer_a_seconds"],
                    "timer_b_seconds": state_machine.session["timer_b_seconds"]
                })

            # Display frame
            cv2.imshow("Two-Man Rule Monitor", frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    print("[SYSTEM] Processing complete.")
    alert_system.save_session_log()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_source = sys.argv[1] if len(sys.argv) > 1 else 0
    main(video_source)
