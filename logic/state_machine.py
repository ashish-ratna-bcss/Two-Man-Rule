# logic/state_machine.py
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import config
from logic.roi_manager import ROIManager
from logic.kinematic_fallback import KinematicFallback

class UnlockState(Enum):
    """Timer states for each lock."""
    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    AUTHORIZED = "authorized"

class DualAuthStateMachine:
    """
    Core state machine for Two-Man Rule dual-control.
    Manages occupancy, timers, ID binding, and authorization checks.
    """

    def __init__(self, roi_manager: ROIManager, fps: int = 30):
        self.roi_manager = roi_manager
        self.fps = fps
        self.dwell_frames = config.calculate_dwell_frames(fps)
        self.fallback = KinematicFallback()

        # Session state
        self.session = config.create_session()

        # Track occupancy
        self.active_ids_in_zone = set()  # Set of track_ids in INTERACTION_ZONE
        self.current_frame_count = 0

    def update_occupancy(self, tracked_persons: Dict[int, Dict]):
        """
        Census update: which track_ids are in INTERACTION_ZONE?
        If count > 2, trigger immediate reset.
        """
        self.active_ids_in_zone = set()

        for track_id, person in tracked_persons.items():
            bbox = person["bbox"]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            if self.roi_manager.point_in_roi("INTERACTION_ZONE", center_x, center_y):
                self.active_ids_in_zone.add(track_id)

        # SECURITY: Overcrowd detection
        if len(self.active_ids_in_zone) > 2:
            self._reset_all_timers()
            self.session["violation_type"] = "OVERCROWD"
            return "VIOLATION_OVERCROWD"

        return "OK"

    def _reset_all_timers(self):
        """Reset all timers and clear session state."""
        self.session["id_a"] = None
        self.session["id_b"] = None
        self.session["timer_a_frames"] = 0
        self.session["timer_b_frames"] = 0
        self.session["timer_a_seconds"] = 0.0
        self.session["timer_b_seconds"] = 0.0
        self.session["grace_buffer_a"] = 0
        self.session["grace_buffer_b"] = 0
        self.session["last_elbow_pos_a"] = None
        self.session["last_elbow_pos_b"] = None
        self.session["violation_type"] = None

    def update_timers(self, tracked_persons: Dict[int, Dict]):
        """
        Update timers for persons at locks.
        Only proceeds if len(active_ids_in_zone) == 2.
        """
        occupancy = len(self.active_ids_in_zone)

        # Logic only proceeds with exactly 2 people
        if occupancy != 2:
            return

        # Try to bind IDs if not yet bound
        if self.session["id_a"] is None:
            available = list(self.active_ids_in_zone)
            self.session["id_a"] = available[0]
            self.session["id_b"] = available[1] if len(available) > 1 else None

        id_a = self.session["id_a"]
        id_b = self.session["id_b"]

        # Update timer for person A at Lock A
        if id_a in tracked_persons:
            self._update_single_timer(
                track_id=id_a,
                person=tracked_persons[id_a],
                lock_roi_name="LOCK_A_ROI",
                timer_key="timer_a_frames",
                timer_seconds_key="timer_a_seconds",
                grace_key="grace_buffer_a"
            )

        # Update timer for person B at Lock B
        if id_b in tracked_persons:
            self._update_single_timer(
                track_id=id_b,
                person=tracked_persons[id_b],
                lock_roi_name="LOCK_B_ROI",
                timer_key="timer_b_frames",
                timer_seconds_key="timer_b_seconds",
                grace_key="grace_buffer_b"
            )

    def _update_single_timer(
        self,
        track_id: int,
        person: Dict,
        lock_roi_name: str,
        timer_key: str,
        timer_seconds_key: str,
        grace_key: str
    ):
        """Update timer for a single person at a single lock."""
        keypoints = person["keypoints"]

        # Determine which wrist/elbow/shoulder based on lock side
        if "A" in lock_roi_name:
            wrist_idx = config.KEYPOINT_WRIST_RIGHT
            elbow_idx = config.KEYPOINT_ELBOW_RIGHT
            shoulder_idx = config.KEYPOINT_SHOULDER_RIGHT
        else:
            wrist_idx = config.KEYPOINT_WRIST_LEFT
            elbow_idx = config.KEYPOINT_ELBOW_LEFT
            shoulder_idx = config.KEYPOINT_SHOULDER_LEFT

        # Get positions
        wrist_x, wrist_y, wrist_conf = keypoints[wrist_idx]
        lock_center = self.roi_manager.get_roi_center(lock_roi_name)

        if not lock_center:
            return

        # Check if wrist is in lock ROI
        wrist_at_lock = self.roi_manager.point_in_roi(lock_roi_name, wrist_x, wrist_y)

        grace_remaining = self.session[grace_key]
        current_timer = self.session[timer_key]

        if wrist_at_lock:
            # Wrist at lock: increment timer
            should_increment = self.fallback.should_continue_timer(
                track_id,
                keypoints,
                wrist_idx,
                elbow_idx,
                shoulder_idx,
                lock_center,
                grace_remaining
            )

            if should_increment:
                self.session[timer_key] = current_timer + 1
                self.session[timer_seconds_key] = self.session[timer_key] / self.fps
                self.session[grace_key] = config.GRACE_BUFFER_FRAMES  # Reset grace buffer
            else:
                # Timer stops
                pass
        else:
            # Wrist NOT at lock
            if grace_remaining > 0:
                # Grace buffer active: continue timer
                self.session[grace_key] = grace_remaining - 1
                self.session[timer_key] = current_timer + 1
                self.session[timer_seconds_key] = self.session[timer_key] / self.fps
            else:
                # Grace expired: reset timer
                self.session[timer_key] = 0
                self.session[timer_seconds_key] = 0.0

    def check_authorization(self) -> Dict[str, any]:
        """
        Check if both locks are authorized.

        Returns:
            {
                "authorized": bool,
                "lock_a_authorized": bool,
                "lock_b_authorized": bool,
                "violation_type": str or None
            }
        """
        timer_a = self.session["timer_a_frames"]
        timer_b = self.session["timer_b_frames"]
        id_a = self.session["id_a"]
        id_b = self.session["id_b"]

        lock_a_auth = timer_a >= self.dwell_frames
        lock_b_auth = timer_b >= self.dwell_frames

        # Check dual-auth: must have exactly 2 IDs and they must be different
        if id_a is None or id_b is None:
            return {
                "authorized": False,
                "lock_a_authorized": lock_a_auth,
                "lock_b_authorized": lock_b_auth,
                "violation_type": None
            }

        if id_a == id_b:
            return {
                "authorized": False,
                "lock_a_authorized": lock_a_auth,
                "lock_b_authorized": lock_b_auth,
                "violation_type": "SAME_ID"
            }

        authorized = lock_a_auth and lock_b_auth and id_a != id_b

        return {
            "authorized": authorized,
            "lock_a_authorized": lock_a_auth,
            "lock_b_authorized": lock_b_auth,
            "violation_type": None if authorized else "INCOMPLETE"
        }

    def get_session(self) -> Dict:
        """Return current session state."""
        return self.session.copy()

    def reset_session(self):
        """Reset session for next cycle."""
        self._reset_all_timers()
        self.current_frame_count += 1
