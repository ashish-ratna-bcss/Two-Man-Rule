# models/tracker.py
from supervision import ByteTrack, Detections
import numpy as np
from typing import Dict, List, Tuple, Optional
import config

class PersonTracker:
    """Enhanced ByteTrack with pose-based ReID: spatial gate + normalized distance."""

    def __init__(self):
        self.tracker = ByteTrack(
            lost_track_buffer=config.TRACK_BUFFER,
            track_activation_threshold=config.TRACK_THRESH
        )
        self.tracked_persons = {}
        self.reid_history = {}
        self.reid_max_history = 15
        self.aliases = {}
        self.lost_id_frames = {}
        self.max_lost_frames = config.TRACK_BUFFER
        self.last_known_bbox = {}  # true_id → last confirmed bbox

    def _compute_keypoint_distance(
        self, kpts1: np.ndarray, kpts2: np.ndarray, bbox_height: float = None
    ) -> float:
        """
        Normalized mean keypoint distance.
        Returns distance / bbox_height so threshold is scale-invariant.
        Requires >= 8 mutually visible keypoints for a reliable match.
        """
        if kpts1 is None or kpts2 is None:
            return float('inf')

        mask = (kpts1[:, 2] > 0.3) & (kpts2[:, 2] > 0.3)
        if np.sum(mask) < 8:
            return float('inf')

        dists = np.sqrt(np.sum((kpts1[mask, :2] - kpts2[mask, :2])**2, axis=1))
        raw_dist = float(np.mean(dists))

        if bbox_height and bbox_height > 0:
            return raw_dist / bbox_height
        return float('inf')  # no scale info → refuse to match

    def _average_keypoints(self, history: List[np.ndarray]) -> np.ndarray:
        """Vectorized averaging of keypoint history."""
        if not history:
            return np.zeros((17, 3), dtype=np.float32)

        hist_arr = np.array(history)  # (N, 17, 3)
        avg_kpts = np.zeros((17, 3), dtype=np.float32)
        for i in range(17):
            points = hist_arr[:, i, :]
            mask = points[:, 2] > 0.3
            if np.any(mask):
                avg_kpts[i, :2] = np.mean(points[mask, :2], axis=0)
                avg_kpts[i, 2] = 1.0
        return avg_kpts

    @staticmethod
    def _bbox_height(bbox) -> float:
        if bbox is None or len(bbox) < 4:
            return 0.0
        return float(bbox[3] - bbox[1])

    @staticmethod
    def _bbox_center_bottom(bbox) -> Optional[Tuple[float, float]]:
        if bbox is None or len(bbox) < 4:
            return None
        return ((bbox[0] + bbox[2]) / 2.0, float(bbox[3]))

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """Update tracker with one-to-one assignment and spatial-gated ReID."""
        if not detections:
            return {}

        det_bboxes = np.array([d["bbox"] for d in detections])
        confidences = np.array([d["confidence"] for d in detections])

        sup_detections = Detections(
            xyxy=det_bboxes,
            confidence=confidences,
            class_id=np.zeros(len(det_bboxes), dtype=int)
        )

        tracked_dets = self.tracker.update_with_detections(sup_detections)
        current_tracked_ids = set()
        det_by_b_id = {}

        if len(tracked_dets) > 0:
            track_bboxes = tracked_dets.xyxy

            # Use bbox center distance (more stable than top-left corner)
            track_centers = np.stack([
                (track_bboxes[:, 0] + track_bboxes[:, 2]) / 2,
                (track_bboxes[:, 1] + track_bboxes[:, 3]) / 2,
            ], axis=1)
            det_centers = np.stack([
                (det_bboxes[:, 0] + det_bboxes[:, 2]) / 2,
                (det_bboxes[:, 1] + det_bboxes[:, 3]) / 2,
            ], axis=1)

            dists = np.linalg.norm(
                track_centers[:, None, :] - det_centers[None, :, :], axis=2
            )  # (T, D)

            # Greedy one-to-one assignment: closest track claims detection first
            used_det = set()
            best_det_indices = {}
            for i in np.argsort(dists.min(axis=1)):
                for det_idx in np.argsort(dists[i]):
                    if det_idx not in used_det:
                        best_det_indices[int(i)] = int(det_idx)
                        used_det.add(det_idx)
                        break

            for i in range(len(tracked_dets)):
                det_idx = best_det_indices.get(i)
                if det_idx is None:
                    continue
                b_id_int = int(tracked_dets.tracker_id[i])
                true_id = self.aliases.get(b_id_int, b_id_int)
                det_by_b_id[b_id_int] = (true_id, detections[det_idx])
                current_tracked_ids.add(true_id)

        # ── ReID: fires immediately (same frame) when ByteTrack drops an ID ──
        lost_ids = [tid for tid in self.reid_history if tid not in current_tracked_ids]
        for lost_id in lost_ids:
            self.lost_id_frames[lost_id] = self.lost_id_frames.get(lost_id, 0) + 1
            if self.lost_id_frames[lost_id] > self.max_lost_frames:
                self.reid_history.pop(lost_id, None)
                self.lost_id_frames.pop(lost_id, None)
                self.last_known_bbox.pop(lost_id, None)
                continue

            history = self.reid_history[lost_id]
            if not history:
                continue

            lost_template = self._average_keypoints(history)
            last_bbox = self.last_known_bbox.get(lost_id)
            last_h = self._bbox_height(last_bbox)
            last_center = self._bbox_center_bottom(last_bbox)

            # Spatial gate: candidate foot must be within 2× body-heights of last position.
            # Grows slightly each lost frame to handle slow drift.
            frames_lost = self.lost_id_frames[lost_id]
            spatial_limit = max(last_h * 2.0, 200.0) + frames_lost * 8.0

            # Normalized threshold: mean keypoint displacement ≤ 35% of body height.
            # Scale-invariant — works for near and far cameras.
            NORM_THRESH = 0.35

            best_b_id, best_dist = None, NORM_THRESH

            for b_id_int, (true_id, det) in det_by_b_id.items():
                if true_id == lost_id:
                    continue

                # Spatial gate — skip candidate if far from where the person was last seen
                if last_center is not None:
                    cand_bbox = det.get("bbox")
                    if cand_bbox is not None:
                        cx = (cand_bbox[0] + cand_bbox[2]) / 2.0
                        cy = float(cand_bbox[3])
                        if np.sqrt((cx - last_center[0])**2 + (cy - last_center[1])**2) > spatial_limit:
                            continue

                cand_h = self._bbox_height(det.get("bbox"))
                ref_h = max(last_h, cand_h) if (last_h > 0 and cand_h > 0) else None
                dist = self._compute_keypoint_distance(lost_template, det.get("keypoints"), ref_h)
                if dist < best_dist:
                    best_dist, best_b_id = dist, b_id_int

            if best_b_id is not None:
                old_true_id = det_by_b_id[best_b_id][0]
                print(f"[REID] RESTORE: {old_true_id} -> {lost_id} "
                      f"(norm_dist={best_dist:.3f}, lost_frames={frames_lost})")
                self.aliases[best_b_id] = lost_id
                det_by_b_id[best_b_id] = (lost_id, det_by_b_id[best_b_id][1])
                self.lost_id_frames[lost_id] = 0
                current_tracked_ids.add(lost_id)

        # Finalize: store keypoints + last bbox for every confirmed track
        new_tracked = {}
        for b_id_int, (true_id, det) in det_by_b_id.items():
            new_tracked[true_id] = {**det, "track_id": true_id}
            self.lost_id_frames[true_id] = 0

            bbox = det.get("bbox")
            if bbox is not None:
                self.last_known_bbox[true_id] = bbox

            curr_kpts = det.get("keypoints")
            if curr_kpts is not None:
                if true_id not in self.reid_history:
                    self.reid_history[true_id] = []
                self.reid_history[true_id].append(curr_kpts)
                if len(self.reid_history[true_id]) > self.reid_max_history:
                    self.reid_history[true_id].pop(0)

        self.tracked_persons = new_tracked
        return self.tracked_persons
