# logic/crop_reid.py
"""Lightweight appearance-based ReID using HSV color histograms.

No external models needed. Extracts upper-body color histogram from bbox crop,
compares via correlation. Works in O(n_detections) per frame.
"""
import cv2
import numpy as np
from typing import Dict, Optional


class CropReID:
    """Save crop features at qualification; re-identify from appearance alone."""

    HIST_BINS = (16, 16, 16)  # H, S, V bins
    UPPER_BODY_FRAC = 0.55    # Use top 55% of bbox (face+torso, skip legs)
    MIN_CROP_SIZE = 20         # Ignore tiny crops

    def __init__(self):
        # slot -> {"hist": np.ndarray, "height": float}
        self._saved = {"a": None, "b": None}

    def save(self, slot: str, frame: np.ndarray, bbox) -> bool:
        """Save appearance feature for slot from current frame."""
        hist = self._extract(frame, bbox)
        if hist is None:
            return False
        h = float(bbox[3] - bbox[1])
        self._saved[slot] = {"hist": hist, "height": h}
        return True

    def find_best_match(
        self,
        slot: str,
        frame: np.ndarray,
        candidates: Dict[int, dict],
        exclude_ids: set = None,
        size_tolerance: float = 0.45,
    ) -> Optional[int]:
        """Return track_id of best appearance match, or None if below threshold."""
        saved = self._saved.get(slot)
        if saved is None:
            return None

        exclude_ids = exclude_ids or set()
        ref_hist = saved["hist"]
        ref_h = saved["height"]

        best_score = 0.55  # Minimum correlation threshold (0..1)
        best_id = None

        for tid, person in candidates.items():
            if tid in exclude_ids:
                continue
            bbox = person.get("bbox")
            if bbox is None:
                continue

            # Size gate: skip if body height differs too much
            cand_h = float(bbox[3] - bbox[1])
            if ref_h > 0 and abs(cand_h - ref_h) / ref_h > size_tolerance:
                continue

            hist = self._extract(frame, bbox)
            if hist is None:
                continue

            score = float(cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL))
            if score > best_score:
                best_score = score
                best_id = tid

        return best_id

    def has_saved(self, slot: str) -> bool:
        return self._saved.get(slot) is not None

    # ── internal ──────────────────────────────────────────────────────────

    def _extract(self, frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        if bbox is None or len(bbox) < 4:
            return None
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        # Use upper body only
        h = y2 - y1
        y2_upper = y1 + int(h * self.UPPER_BODY_FRAC)
        crop = frame[y1:y2_upper, x1:x2]

        if crop.shape[0] < self.MIN_CROP_SIZE or crop.shape[1] < self.MIN_CROP_SIZE:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, list(self.HIST_BINS),
            [0, 180, 0, 256, 0, 256]
        )
        cv2.normalize(hist, hist)
        return hist.flatten()
