# models/tracker.py
from supervision import ByteTrack, Detections
import numpy as np
from typing import Dict, List, Tuple, Optional
import config

class PersonTracker:
    """Enhanced ByteTrack with pose-based ReID for persistent ID assignment (StrongSORT-like)."""

    def __init__(self):
        """Initialize ByteTrack with ReID embedding history."""
        self.tracker = ByteTrack(
            lost_track_buffer=config.TRACK_BUFFER,
            track_activation_threshold=config.TRACK_THRESH
        )
        self.tracked_persons = {}  # track_id -> person_data
        # ReID history: track_id -> list of recent pose embeddings
        self.reid_history = {}
        self.reid_max_history = 15  # Keep last 15 embeddings (e.g. 0.5s at 30fps) per track
        self.aliases = {}  # ByteTrack ID -> ReID mapped ID
        self.lost_id_frames = {}  # track_id -> frames since last seen
        self.max_lost_frames = config.TRACK_BUFFER  # Purge after this many frames (from config)

    def _compute_keypoint_distance(self, kpts1: np.ndarray, kpts2: np.ndarray) -> float:
        """Compute average Euclidean distance between corresponding visible actual keypoints."""
        if kpts1 is None or kpts2 is None:
            return float('inf')

        # kpts: (17, 3) where 3 is (x, y, conf)
        vis1 = kpts1[:, 2] > 0.3
        vis2 = kpts2[:, 2] > 0.3
        mask = vis1 & vis2

        if mask.sum() < 5:  # Need at least 5 common keypoints
            return float('inf')

        diff = kpts1[mask, :2] - kpts2[mask, :2]
        dists = np.linalg.norm(diff, axis=1)
        return float(np.mean(dists))

    def _average_keypoints(self, history: List[np.ndarray]) -> np.ndarray:
        """Create a single template keypoints array from a history of frames."""
        avg_kpts = np.zeros((17, 3), dtype=np.float32)

        for i in range(17):
            valid_pts = []
            for kpts in history:
                if kpts is not None and len(kpts) > i and kpts[i, 2] > 0.3:
                    valid_pts.append(kpts[i, :2])

            if valid_pts:
                avg_kpts[i, :2] = np.mean(valid_pts, axis=0)
                avg_kpts[i, 2] = 1.0  # Mark as valid
            else:
                avg_kpts[i, 2] = 0.0  # Mark as invalid

        return avg_kpts

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracker with ReID-enhanced ByteTrack mapping.
        """
        if not detections:
            return {}

        bboxes = np.array([d["bbox"] for d in detections])
        confidences = np.array([d["confidence"] for d in detections])

        sup_detections = Detections(
            xyxy=bboxes,
            confidence=confidences,
            class_id=np.zeros(len(bboxes), dtype=int)
        )

        # Run ByteTrack
        tracked_dets = self.tracker.update_with_detections(sup_detections)

        current_tracked_ids = set()
        det_by_b_id = {}

        # 1. Match ByteTrack output back to original detections (for keypoints)
        for i in range(len(tracked_dets)):
            b_id_int = int(tracked_dets.tracker_id[i])
            t_box = tracked_dets.xyxy[i]

            best_det = None
            best_dist = float('inf')
            for d in detections:
                d_box = d["bbox"]
                # Match by top-left coordinate distance
                dist = np.linalg.norm(np.array(t_box[:2]) - np.array(d_box[:2]))
                if dist < best_dist:
                    best_dist = dist
                    best_det = d

            if best_det is None:
                continue

            # Resolve Alias
            true_id = self.aliases.get(b_id_int, b_id_int)
            det_by_b_id[b_id_int] = (true_id, best_det)
            current_tracked_ids.add(true_id)

        # 2. ReID verification for "Lost" IDs (Tracks in history but missing from current scene)
        lost_ids = [tid for tid in self.reid_history.keys() if tid not in current_tracked_ids]

        for lost_id in lost_ids:
            # Increment lost counter
            self.lost_id_frames[lost_id] = self.lost_id_frames.get(lost_id, 0) + 1

            # If person is gone for too long, purge fingerprint to avoid false matches later
            if self.lost_id_frames[lost_id] > self.max_lost_frames:
                if lost_id in self.reid_history:
                    del self.reid_history[lost_id]
                if lost_id in self.lost_id_frames:
                    del self.lost_id_frames[lost_id]
                continue

            history = self.reid_history[lost_id]
            if not history:
                continue

            lost_template_kpts = self._average_keypoints(history)

            best_b_id = None
            # Threshold: maximum average pixel distance to be considered the same person
            best_dist = 150.0

            # Check all persons in the full scene to see if anyone matches this STICKY fingerprint
            for b_id_int, (true_id, det) in det_by_b_id.items():
                # Don't try to assign the same ID to two different people
                if true_id == lost_id:
                    continue

                curr_kpts = det.get("keypoints")
                dist = self._compute_keypoint_distance(lost_template_kpts, curr_kpts)

                if dist < best_dist:
                    best_dist = dist
                    best_b_id = b_id_int

            if best_b_id is not None:
                # We found someone near the lost ID's location! Reassign ID.
                # Remove alias from old ID and set to lost_id
                old_true_id = det_by_b_id[best_b_id][0]

                print(f"[REID] ID RESTORE: Lost ID {lost_id} found! Reassigning Scene ID {old_true_id} -> {lost_id} (dist={best_dist:.1f}px)")

                self.aliases[best_b_id] = lost_id
                det_by_b_id[best_b_id] = (lost_id, det_by_b_id[best_b_id][1])

                # Reset lost counter for recovered ID
                self.lost_id_frames[lost_id] = 0
                current_tracked_ids.add(lost_id)
                if old_true_id in current_tracked_ids and old_true_id != lost_id:
                    current_tracked_ids.remove(old_true_id)

        # 3. Finalize and Store Keypoint Arrays (Last stack of arrays)
        new_tracked = {}
        for b_id_int, (true_id, det) in det_by_b_id.items():
            person_data = {**det, "track_id": true_id}
            new_tracked[true_id] = person_data

            # Reset lost counter since they are active
            self.lost_id_frames[true_id] = 0

            curr_kpts = det.get("keypoints")
            if curr_kpts is not None:
                if true_id not in self.reid_history:
                    self.reid_history[true_id] = []
                self.reid_history[true_id].append(curr_kpts)
                if len(self.reid_history[true_id]) > self.reid_max_history:
                    self.reid_history[true_id].pop(0)

        self.tracked_persons = new_tracked
        return self.tracked_persons

    def get_tracked_ids(self) -> List[int]:
        """Return list of currently tracked person IDs."""
        return list(self.tracked_persons.keys())

    def get_person(self, track_id: int) -> Dict:
        """Get tracked person data by ID."""
        return self.tracked_persons.get(track_id)
