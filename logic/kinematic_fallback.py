# logic/kinematic_fallback.py
import numpy as np
from typing import Dict, Tuple, Optional
import config

class KinematicFallback:
    """Handle wrist occlusion via elbow proximity + shoulder stability."""

    def __init__(self):
        self.shoulder_history = {}  # track_id -> [(x, y), ...]
        self.max_history = 5  # Frames to check for stillness

    def is_wrist_visible(self, keypoints: np.ndarray, wrist_idx: int) -> bool:
        """Check if wrist keypoint has sufficient confidence."""
        _, _, conf = keypoints[wrist_idx]
        return conf >= config.WRIST_CONFIDENCE_THRESHOLD

    def get_position(self, keypoints: np.ndarray, keypoint_idx: int) -> Optional[Tuple[float, float]]:
        """Extract (x, y) from keypoint array."""
        if keypoint_idx >= len(keypoints):
            return None
        x, y, conf = keypoints[keypoint_idx]
        return (float(x), float(y)) if conf > 0 else None

    def distance_euclidean(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def is_shoulder_static(self, track_id: int, shoulder_pos: Tuple[float, float], threshold_px: float = 3.0) -> bool:
        """
        Check if shoulder position is stable across recent frames.
        Returns True if position variance is below threshold.
        """
        if track_id not in self.shoulder_history:
            self.shoulder_history[track_id] = []

        history = self.shoulder_history[track_id]
        history.append(shoulder_pos)

        # Keep only last N frames
        if len(history) > self.max_history:
            history.pop(0)

        # Need at least 2 frames to check
        if len(history) < 2:
            return True  # Assume static if not enough history

        # Check max distance between any two points
        max_distance = 0
        for i in range(len(history)):
            for j in range(i + 1, len(history)):
                dist = self.distance_euclidean(history[i], history[j])
                max_distance = max(max_distance, dist)

        return max_distance < threshold_px

    def should_continue_timer(
        self,
        track_id: int,
        keypoints: np.ndarray,
        wrist_idx: int,
        elbow_idx: int,
        shoulder_idx: int,
        lock_roi_center: Tuple[float, float],
        grace_buffer_remaining: int
    ) -> bool:
        """
        Determine if 10-second timer should continue despite wrist occlusion.

        Returns:
            True if wrist is visible OR (elbow is close AND shoulder is static)
        """
        wrist_visible = self.is_wrist_visible(keypoints, wrist_idx)

        # If wrist is visible, always continue
        if wrist_visible:
            return True

        # Wrist not visible - check grace buffer
        if grace_buffer_remaining > 0:
            return True

        # Grace buffer exhausted - check elbow proximity + shoulder stability
        elbow_pos = self.get_position(keypoints, elbow_idx)
        shoulder_pos = self.get_position(keypoints, shoulder_idx)

        if not elbow_pos or not shoulder_pos:
            return False

        # Elbow close to lock?
        elbow_distance = self.distance_euclidean(elbow_pos, lock_roi_center)
        elbow_close = elbow_distance <= config.ELBOW_LOCK_PROXIMITY_PIXELS

        # Shoulder static?
        shoulder_stable = self.is_shoulder_static(track_id, shoulder_pos)

        return elbow_close and shoulder_stable

    def reset_track_history(self, track_id: int):
        """Clear history for a track (e.g., when track is lost)."""
        if track_id in self.shoulder_history:
            del self.shoulder_history[track_id]
