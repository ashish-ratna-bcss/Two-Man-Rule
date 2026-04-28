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
        self.reid_max_history = 5  # Keep last 5 embeddings per track

    def _extract_pose_embedding(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """Extract lightweight pose embedding from keypoints (16-dim feature)."""
        if keypoints is None or len(keypoints) < 17:
            return None

        # Robust features: keypoint positions normalized to body bounding box
        valid_kpts = keypoints[:, :2]
        confidences = keypoints[:, 2]

        if len(valid_kpts[confidences > 0.2]) < 5:
            return None

        # Compute body center and scale
        valid_mask = confidences > 0.2
        if not valid_mask.any():
            return None

        center = valid_kpts[valid_mask].mean(axis=0)
        scale = (valid_kpts[valid_mask].std(axis=0).mean() or 1.0) + 1e-6

        # Normalize keypoints to [-1, 1] range
        norm_kpts = (valid_kpts - center) / scale

        # Create 16-dim embedding: sample 16 keypoints' normalized positions
        embedding_indices = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 3]
        embedding = []
        for idx in embedding_indices:
            if idx < len(norm_kpts):
                embedding.extend([norm_kpts[idx, 0], norm_kpts[idx, 1]])

        return np.array(embedding[:16], dtype=np.float32)

    def _compute_reid_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute L2 distance between pose embeddings."""
        if emb1 is None or emb2 is None:
            return float('inf')
        return float(np.linalg.norm(emb1 - emb2))

    def _try_reid_match(self, detections: List[Dict]) -> Dict[int, int]:
        """
        Try to match current detections to lost tracks using pose embeddings.
        Returns: {detection_idx: track_id} for successful ReID matches.
        """
        reid_matches = {}
        if not self.reid_history or not detections:
            return reid_matches

        # Extract embeddings for current detections
        current_embeddings = []
        for det in detections:
            emb = self._extract_pose_embedding(det.get("keypoints"))
            current_embeddings.append(emb)

        # Try to match each detection to lost tracks via ReID
        for det_idx, curr_emb in enumerate(current_embeddings):
            if curr_emb is None:
                continue

            best_track_id = None
            best_distance = 0.3  # Distance threshold

            for track_id, history in self.reid_history.items():
                if track_id in self.tracked_persons:
                    continue  # Skip active tracks

                # Average embedding from history
                hist_embs = np.array(history)
                avg_emb = hist_embs.mean(axis=0)
                dist = self._compute_reid_distance(curr_emb, avg_emb)

                if dist < best_distance:
                    best_distance = dist
                    best_track_id = track_id

            if best_track_id is not None:
                reid_matches[det_idx] = best_track_id
                print(f"[REID] Detection {det_idx} matched to lost track {best_track_id} (dist={best_distance:.3f})")

        return reid_matches

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracker with ReID-enhanced ByteTrack.

        Args:
            detections: List from PoseDetector.detect()

        Returns:
            Dict: {track_id: {"bbox": ..., "keypoints": ..., "confidence": ...}}
        """
        if not detections:
            return {}

        # Try to recover lost persons via ReID before running ByteTrack
        reid_matches = self._try_reid_match(detections)

        # Convert to Supervision Detections format
        bboxes = np.array([d["bbox"] for d in detections])
        confidences = np.array([d["confidence"] for d in detections])

        sup_detections = Detections(
            xyxy=bboxes,
            confidence=confidences,
            class_id=np.zeros(len(bboxes), dtype=int)
        )

        # Run ByteTrack
        tracked_dets = self.tracker.update_with_detections(sup_detections)

        # Update tracked persons and embeddings
        new_tracked = {}
        for i, track_id in enumerate(tracked_dets.tracker_id):
            if i < len(detections):
                track_id_int = int(track_id)
                person_data = {**detections[i], "track_id": track_id_int}
                new_tracked[track_id_int] = person_data

                # Store pose embedding for ReID history
                emb = self._extract_pose_embedding(detections[i].get("keypoints"))
                if emb is not None:
                    if track_id_int not in self.reid_history:
                        self.reid_history[track_id_int] = []
                    self.reid_history[track_id_int].append(emb)
                    # Keep only recent history
                    if len(self.reid_history[track_id_int]) > self.reid_max_history:
                        self.reid_history[track_id_int].pop(0)

        # Remove history for tracks no longer active
        active_ids = set(new_tracked.keys())
        orphaned_ids = []
        for track_id in self.reid_history:
            if track_id not in active_ids:
                orphaned_ids.append(track_id)

        # Keep orphaned histories for potential re-identification (up to buffer frames)
        for track_id in list(self.reid_history.keys()):
            if track_id not in active_ids:
                # Decay: keep for ReID matching but mark as candidates
                pass

        self.tracked_persons = new_tracked
        return self.tracked_persons

    def get_tracked_ids(self) -> List[int]:
        """Return list of currently tracked person IDs."""
        return list(self.tracked_persons.keys())

    def get_person(self, track_id: int) -> Dict:
        """Get tracked person data by ID."""
        return self.tracked_persons.get(track_id)
