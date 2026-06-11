# models/tracker.py
from collections import deque
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
        # Reverse index: true_id → [b_id_int, ...] for O(1) alias cleanup on track expiry.
        # Without this, aliases accumulates every Re-ID ever made and never shrinks.
        self._alias_reverse: dict = {}
        self.lost_id_frames = {}
        self.max_lost_frames = config.TRACK_BUFFER
        self.last_known_bbox = {}  # true_id → last confirmed bbox
        # Deep appearance Re-ID: per-track L2-normalized embedding template (EMA).
        self.embedding_gallery = {}  # true_id → np.ndarray(D,)
        self._reid_enabled = bool(getattr(config, "REID_APPEARANCE_ENABLED", False))
        self._reid_cos_max = float(getattr(config, "REID_COSINE_MAX", 0.40))
        self._reid_ema = float(getattr(config, "REID_GALLERY_EMA", 0.9))

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
    def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance in [0, 2] for L2-normalized vectors (0 = identical)."""
        return 1.0 - float(np.dot(a, b))

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

    def update(self, detections: List[Dict], protected_ids: set = None,
               embeddings=None) -> Dict[int, Dict]:
        """
        Update tracker with one-to-one assignment and spatial-gated ReID.

        Args:
            detections: List of detection dicts with bbox, confidence, keypoints.
            protected_ids: Set of true_ids that must NOT be stolen by Re-ID and
                assigned to a different physical person (e.g. verified unlocker IDs).
                These IDs are still tracked normally when directly detected — protection
                only blocks a *lost* protected ID from being matched to a new passer-by.
            embeddings: Optional (N, D) array of L2-normalized appearance embeddings,
                aligned to ``detections``. When present, lost-id Re-ID ranks candidates
                by embedding cosine distance (cross-view, gap-robust) instead of
                pose-keypoint distance. Falls back to keypoints when absent/zero.
        """
        if not detections:
            return {}

        # Attach per-detection embedding (None when unavailable or a zero vector).
        if embeddings is not None and len(embeddings) == len(detections):
            for i, d in enumerate(detections):
                emb = embeddings[i]
                d["_embedding"] = (
                    emb if (emb is not None and getattr(emb, "size", 0) > 0
                            and float(np.linalg.norm(emb)) > 0.0)
                    else None
                )
        else:
            for d in detections:
                d.setdefault("_embedding", None)

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
        _protected = protected_ids or set()
        # reid_claimed: b_id_int slots already consumed by a Re-ID match this frame.
        # Prevents within-frame chaining where stealing b_id X displaces its previous
        # true_id, which then immediately steals b_id Y, etc. — creating 3-way loops
        # (e.g. 10→3→5→10 each frame) that eventually starve a candidate's true_id.
        reid_claimed = set()
        lost_ids = [tid for tid in self.reid_history if tid not in current_tracked_ids]
        for lost_id in lost_ids:
            self.lost_id_frames[lost_id] = self.lost_id_frames.get(lost_id, 0) + 1
            if self.lost_id_frames[lost_id] > self.max_lost_frames:
                self.reid_history.pop(lost_id, None)
                self.lost_id_frames.pop(lost_id, None)
                self.last_known_bbox.pop(lost_id, None)
                self.embedding_gallery.pop(lost_id, None)
                # Remove all aliases pointing to this expired true_id (O(1) via reverse index).
                for b_id in self._alias_reverse.pop(lost_id, []):
                    self.aliases.pop(b_id, None)
                continue

            # Skip Re-ID for protected IDs (verified + candidate unlockers): the
            # state machine's _remap_verified_unlocker handles recovery at a higher,
            # context-aware level and should not be pre-empted by the tracker.
            if lost_id in _protected:
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

            # Prefer deep appearance matching when this lost id has an embedding
            # template; it re-links across angle/gap where keypoints cannot. Fall back
            # to pose-keypoint distance otherwise.
            lost_emb = self.embedding_gallery.get(lost_id) if self._reid_enabled else None
            use_emb = lost_emb is not None
            best_b_id = None
            best_dist = self._reid_cos_max if use_emb else NORM_THRESH
            match_kind = "cos" if use_emb else "norm"

            for b_id_int, (true_id, det) in det_by_b_id.items():
                if true_id == lost_id:
                    continue
                # Skip slots already consumed by a Re-ID match this frame
                if b_id_int in reid_claimed:
                    continue
                # Skip slots whose current holder is a protected ID — displacing them
                # would cause the protected person's detection to vanish from reid_history.
                if true_id in _protected:
                    continue

                # Spatial gate — skip candidate if far from where the person was last seen
                if last_center is not None:
                    cand_bbox = det.get("bbox")
                    if cand_bbox is not None:
                        cx = (cand_bbox[0] + cand_bbox[2]) / 2.0
                        cy = float(cand_bbox[3])
                        if np.sqrt((cx - last_center[0])**2 + (cy - last_center[1])**2) > spatial_limit:
                            continue

                if use_emb:
                    cand_emb = det.get("_embedding")
                    if cand_emb is None:
                        continue
                    cos_dist = self._cos_dist(lost_emb, cand_emb)
                    if cos_dist < best_dist:
                        best_dist, best_b_id = cos_dist, b_id_int
                else:
                    cand_h = self._bbox_height(det.get("bbox"))
                    ref_h = max(last_h, cand_h) if (last_h > 0 and cand_h > 0) else None
                    dist = self._compute_keypoint_distance(lost_template, det.get("keypoints"), ref_h)
                    if dist < best_dist:
                        best_dist, best_b_id = dist, b_id_int

            if best_b_id is not None:
                old_true_id = det_by_b_id[best_b_id][0]
                print(f"[REID] RESTORE: {old_true_id} -> {lost_id} "
                      f"({match_kind}_dist={best_dist:.3f}, lost_frames={frames_lost})")
                # Keep reverse index in sync: remove previous mapping for this b_id slot.
                if best_b_id in self.aliases:
                    prev_tid = self.aliases[best_b_id]
                    refs = self._alias_reverse.get(prev_tid)
                    if refs is not None:
                        try:
                            refs.remove(best_b_id)
                        except ValueError:
                            pass
                self.aliases[best_b_id] = lost_id
                self._alias_reverse.setdefault(lost_id, []).append(best_b_id)
                det_by_b_id[best_b_id] = (lost_id, det_by_b_id[best_b_id][1])
                self.lost_id_frames[lost_id] = 0
                current_tracked_ids.add(lost_id)
                reid_claimed.add(best_b_id)  # lock this slot for the rest of the frame

        # Finalize: store keypoints + last bbox for every confirmed track
        new_tracked = {}
        for b_id_int, (true_id, det) in det_by_b_id.items():
            new_tracked[true_id] = {**det, "track_id": true_id}
            self.lost_id_frames[true_id] = 0

            bbox = det.get("bbox")
            if bbox is not None:
                self.last_known_bbox[true_id] = bbox

            # Maintain the EMA appearance template for this track.
            emb = det.get("_embedding")
            if self._reid_enabled and emb is not None:
                prev = self.embedding_gallery.get(true_id)
                if prev is None:
                    self.embedding_gallery[true_id] = emb.copy()
                else:
                    merged = self._reid_ema * prev + (1.0 - self._reid_ema) * emb
                    n = float(np.linalg.norm(merged))
                    self.embedding_gallery[true_id] = (merged / n) if n > 0 else emb.copy()

            curr_kpts = det.get("keypoints")
            if curr_kpts is not None:
                if true_id not in self.reid_history:
                    self.reid_history[true_id] = deque(maxlen=self.reid_max_history)
                self.reid_history[true_id].append(curr_kpts)

        self.tracked_persons = new_tracked
        return self.tracked_persons

    def reset(self) -> None:
        """Explicit full reset — call at window end before process exit.
        Releases all numpy arrays and track state without relying on GC.
        """
        self.tracked_persons.clear()
        self.reid_history.clear()
        self.aliases.clear()
        self._alias_reverse.clear()
        self.lost_id_frames.clear()
        self.last_known_bbox.clear()
        self.embedding_gallery.clear()
