# models/tracker.py
from supervision import ByteTrack, Detections
import numpy as np
from typing import Dict, List, Tuple
import config

class PersonTracker:
    """ByteTrack wrapper for persistent person ID assignment."""

    def __init__(self):
        """Initialize ByteTrack."""
        self.tracker = ByteTrack(
            track_buffer=config.TRACK_BUFFER,
            track_thresh=config.TRACK_THRESH
        )
        self.tracked_persons = {}  # track_id -> person_data

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracker with new detections.

        Args:
            detections: List from PoseDetector.detect()

        Returns:
            Dict: {track_id: {"bbox": ..., "keypoints": ..., "confidence": ...}}
        """
        if not detections:
            return {}

        # Convert to Supervision Detections format
        bboxes = np.array([d["bbox"] for d in detections])
        confidences = np.array([d["confidence"] for d in detections])

        # Create Detections object (xyxy format)
        sup_detections = Detections(
            xyxy=bboxes,
            confidence=confidences,
            class_id=np.zeros(len(bboxes), dtype=int)  # All persons (class 0)
        )

        # Run ByteTrack
        tracked_dets = self.tracker.update_with_detections(sup_detections)

        # Map back to original detection data with track IDs
        self.tracked_persons = {}
        for i, track_id in enumerate(tracked_dets.tracker_id):
            if i < len(detections):
                self.tracked_persons[int(track_id)] = {
                    **detections[i],
                    "track_id": int(track_id)
                }

        return self.tracked_persons

    def get_tracked_ids(self) -> List[int]:
        """Return list of currently tracked person IDs."""
        return list(self.tracked_persons.keys())

    def get_person(self, track_id: int) -> Dict:
        """Get tracked person data by ID."""
        return self.tracked_persons.get(track_id)
