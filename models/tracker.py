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

    def _extract_pose_embedding(self, keypoints: np.ndarray) -> Optional[dict]:
        """Extract biomechanical fingerprint: normalized relational distances between keypoints."""
        if keypoints is None or len(keypoints) < 17:
            return None

        valid_kpts = keypoints[:, :2]
        confidences = keypoints[:, 2]

        # Require at least some confident keypoints
        vis_mask = confidences > 0.3
        if vis_mask.sum() < 5:
            return None

        # 1. Compute stable normalization metric (torso height)
        norm_scale = 1.0
        s_vis = vis_mask[5] and vis_mask[6]
        h_vis = vis_mask[11] and vis_mask[12]
        if s_vis and h_vis:
            shoulder_mid = (valid_kpts[5] + valid_kpts[6]) / 2.0
            hip_mid = (valid_kpts[11] + valid_kpts[12]) / 2.0
            norm_scale = float(np.linalg.norm(shoulder_mid - hip_mid))
        else:
            # Fallback: max distance between any two visible keypoints
            visible_pts = valid_kpts[vis_mask]
            max_dist = 0
            for i in range(len(visible_pts)):
                for j in range(i+1, len(visible_pts)):
                    d = np.linalg.norm(visible_pts[i] - visible_pts[j])
                    if d > max_dist:
                        max_dist = d
            norm_scale = float(max_dist)

        if norm_scale < 1e-6:
            return None

        # 2. Compute 136 pairwise distances
        n = 17
        distances = []
        valid_pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                if vis_mask[i] and vis_mask[j]:
                    dist = float(np.linalg.norm(valid_kpts[i] - valid_kpts[j])) / norm_scale
                    distances.append(dist)
                    valid_pairs.append(True)
                else:
                    distances.append(0.0)
                    valid_pairs.append(False)

        return {
            "dists": np.array(distances, dtype=np.float32),
            "mask": np.array(valid_pairs, dtype=bool)
        }

    def _compute_reid_distance(self, emb1: dict, emb2: dict) -> float:
        """Compute cosine distance between biomechanical fingerprints (visible pairs only)."""
        if emb1 is None or emb2 is None:
            return float('inf')

        mask = emb1["mask"] & emb2["mask"]
        if mask.sum() < 10:  # Need at least 10 mutual valid pairs to compare
            return float('inf')

        v1 = emb1["dists"][mask]
        v2 = emb2["dists"][mask]

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return float('inf')

        sim = np.dot(v1, v2) / (norm1 * norm2)
        return float(max(0.0, 1.0 - sim))  # Cosine distance

    def _average_embeddings(self, history: List[dict]) -> dict:
        """Create a single template fingerprint from a history of frames."""
        all_dists = np.vstack([h["dists"] for h in history])
        all_masks = np.vstack([h["mask"] for h in history])

        avg_dists = np.zeros(136, dtype=np.float32)
        avg_mask = np.zeros(136, dtype=bool)

        valid_counts = all_masks.sum(axis=0)

        for i in range(136):
            if valid_counts[i] > 0:
                avg_dists[i] = all_dists[:, i][all_masks[:, i]].mean()
                avg_mask[i] = True

        return {"dists": avg_dists, "mask": avg_mask}

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

            lost_template = self._average_embeddings(history)

            best_b_id = None
            # STRICT REQUIREMENT: Confidence > 95% (Distance < 0.05)
            best_dist = 0.05

            # Check all persons in the full scene to see if anyone matches this STICKY fingerprint
            for b_id_int, (true_id, det) in det_by_b_id.items():
                # Don't try to assign the same ID to two different people
                if true_id == lost_id:
                    continue

                curr_emb = self._extract_pose_embedding(det.get("keypoints"))
                dist = self._compute_reid_distance(lost_template, curr_emb)

                if dist < best_dist:
                    best_dist = dist
                    best_b_id = b_id_int

            if best_b_id is not None:
                # We found someone >90% similar! Reassign ID.
                # Remove alias from old ID and set to lost_id
                old_true_id = det_by_b_id[best_b_id][0]

                print(f"[REID] ID RESTORE: Lost ID {lost_id} found! Reassigning Scene ID {old_true_id} -> {lost_id} (conf={(1.0-best_dist)*100:.1f}%)")

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

            emb = self._extract_pose_embedding(det.get("keypoints"))
            if emb is not None:
                if true_id not in self.reid_history:
                    self.reid_history[true_id] = []
                self.reid_history[true_id].append(emb)
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
