from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import cv2
import numpy as np

import config


class FrameQualityStatus(str, Enum):
    GOOD = "GOOD"
    SUSPECT = "SUSPECT"
    CORRUPT = "CORRUPT"
    STALE = "STALE"


@dataclass
class FrameQualityResult:
    status: FrameQualityStatus
    reason: str = "ok"
    usable: bool = True
    degraded: bool = False
    consecutive_bad: int = 0
    consecutive_good: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)


class FrameQualityGate:
    """
    Screens decoded frames before they can update AI, door, or auth state.

    The gate intentionally prefers freezing a few frames over allowing obvious
    decode artifacts to become security events.
    """

    def __init__(
        self,
        door_corner_roi: Optional[np.ndarray] = None,
        *,
        enabled: Optional[bool] = None,
        degraded_after_frames: Optional[int] = None,
        recovery_good_frames: Optional[int] = None,
        stale_after_frames: Optional[int] = None,
    ):
        self.enabled = bool(config.FRAME_QUALITY_ENABLED if enabled is None else enabled)
        self.degraded_after_frames = int(
            degraded_after_frames
            if degraded_after_frames is not None
            else config.FRAME_QUALITY_DEGRADED_AFTER_FRAMES
        )
        self.recovery_good_frames = int(
            recovery_good_frames
            if recovery_good_frames is not None
            else config.FRAME_QUALITY_RECOVERY_GOOD_FRAMES
        )
        self.stale_after_frames = int(
            stale_after_frames
            if stale_after_frames is not None
            else config.FRAME_QUALITY_STALE_AFTER_FRAMES
        )
        self.door_corner_roi = door_corner_roi.reshape(-1, 2).astype(np.int32) if door_corner_roi is not None else None

        self.recovery_good_frames_min = int(
            getattr(config, "FRAME_QUALITY_RECOVERY_GOOD_FRAMES_MIN", self.recovery_good_frames)
        )
        self.recovery_storm_bad_frames = int(
            getattr(config, "FRAME_QUALITY_RECOVERY_STORM_BAD_FRAMES", 60)
        )
        self._cell_grid = tuple(getattr(config, "FRAME_QUALITY_CELL_GRID", (8, 6)))
        self._cell_white_ratio = float(getattr(config, "FRAME_QUALITY_CELL_WHITE_RATIO", 0.85))
        self._flat_white_fraction = float(getattr(config, "FRAME_QUALITY_FLAT_WHITE_FRACTION", 0.03))

        self._last_gray_small = None
        self._last_good_gray_small = None
        self._stale_frames = 0
        self._consecutive_bad = 0
        self._consecutive_good = 0
        self._recovering = False

    def evaluate(self, frame: np.ndarray) -> FrameQualityResult:
        if not self.enabled:
            return FrameQualityResult(status=FrameQualityStatus.GOOD, usable=True)

        status, reason, metrics, gray_small = self._classify(frame)

        if status == FrameQualityStatus.GOOD:
            self._consecutive_good += 1
            # Adaptive recovery: a long freeze storm (large _consecutive_bad, which is held
            # through recovery until it completes) can never reach the full count, so lower
            # the bar to the floor. Each counted frame already passed every corruption check.
            effective_recovery = self.recovery_good_frames
            if self._consecutive_bad >= self.recovery_storm_bad_frames:
                effective_recovery = self.recovery_good_frames_min
            if self._recovering and self._consecutive_good < effective_recovery:
                usable = False
                reason = f"recovering_good_frames_{self._consecutive_good}_of_{effective_recovery}"
            else:
                usable = True
                self._recovering = False
                self._consecutive_bad = 0
            if gray_small is not None:
                self._last_good_gray_small = gray_small.copy()
        else:
            self._consecutive_bad += 1
            self._consecutive_good = 0
            self._recovering = True
            usable = False

        if gray_small is not None:
            self._last_gray_small = gray_small.copy()

        degraded = self._consecutive_bad >= self.degraded_after_frames
        return FrameQualityResult(
            status=status,
            reason=reason,
            usable=usable,
            degraded=degraded,
            consecutive_bad=self._consecutive_bad,
            consecutive_good=self._consecutive_good,
            metrics=metrics,
        )

    def _classify(self, frame: np.ndarray):
        metrics: Dict[str, float] = {}

        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            return FrameQualityStatus.CORRUPT, "invalid_frame_shape", metrics, None

        height, width = frame.shape[:2]

        if height < 16 or width < 16:
            return FrameQualityStatus.CORRUPT, "frame_too_small", metrics, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
        hsv_small = cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)

        metrics.update(self._patch_metrics(gray_small, hsv_small, prefix="frame"))

        if self._last_gray_small is not None:
            frame_delta = float(np.mean(cv2.absdiff(gray_small, self._last_gray_small)))

            print(f"<<<<<<<<<<<<<<< frame_delta: {frame_delta} >>>>>>>>>>>>>>>>>>>>")

            metrics["frame_delta"] = frame_delta

            if frame_delta < 0.2:
                self._stale_frames += 1
            else:
                self._stale_frames = 0

            if self._stale_frames >= self.stale_after_frames:
                return FrameQualityStatus.STALE, "repeated_identical_frames", metrics, gray_small
        else:
            self._stale_frames = 0

        if self._last_good_gray_small is not None:
            good_delta_img = cv2.absdiff(gray_small, self._last_good_gray_small)
            good_delta = float(np.mean(good_delta_img))
            good_delta_std = float(np.std(good_delta_img))
            metrics["last_good_delta"] = good_delta
            metrics["last_good_delta_std"] = good_delta_std
            if good_delta > 95.0 and good_delta_std > 40.0:
                return FrameQualityStatus.SUSPECT, "sudden_spatial_frame_change", metrics, gray_small

        print(f"<<<<<<<<<<<<<<< metrics: {metrics} >>>>>>>>>>>>>>>>>>>>")

        
        frame_reason = self._artifact_reason(metrics, "frame")

        if frame_reason:
            return FrameQualityStatus.CORRUPT, frame_reason, metrics, gray_small

        if self.door_corner_roi is not None:
            crop = self._crop_roi(frame, self.door_corner_roi)
            if crop is not None:
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crop_gray_small = cv2.resize(crop_gray, (32, 24), interpolation=cv2.INTER_AREA)
                crop_hsv_small = cv2.cvtColor(cv2.resize(crop, (32, 24), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
                metrics.update(self._patch_metrics(crop_gray_small, crop_hsv_small, prefix="door"))
                door_reason = self._artifact_reason(metrics, "door")
                if door_reason:
                    return FrameQualityStatus.CORRUPT, door_reason, metrics, gray_small

        return FrameQualityStatus.GOOD, "ok", metrics, gray_small

    def _patch_metrics(self, gray_patch: np.ndarray, hsv_patch: np.ndarray, *, prefix: str) -> Dict[str, float]:
        h, w = gray_patch.shape[:2]
        left = gray_patch[:, : max(1, w // 2)]
        right = gray_patch[:, max(1, w // 2):]
        white_ratio = float(np.mean(gray_patch >= 245))
        black_ratio = float(np.mean(gray_patch <= 8))
        sat = hsv_patch[:, :, 1]
        sat_ratio = float(np.mean(sat >= 120))
        sat_std = float(np.std(sat))
        left_mean = float(np.mean(left))
        right_mean = float(np.mean(right))

        cell_white_max = self._cell_white_max(gray_patch)

        flat_white_frac = self._flat_white_frac(gray_patch)

        return {
            f"{prefix}_white_ratio": white_ratio,
            f"{prefix}_black_ratio": black_ratio,
            f"{prefix}_sat_ratio": sat_ratio,
            f"{prefix}_sat_std": sat_std,
            f"{prefix}_half_mean_delta": abs(left_mean - right_mean),
            f"{prefix}_max_half_mean": max(left_mean, right_mean),
            f"{prefix}_min_half_mean": min(left_mean, right_mean),
            f"{prefix}_cell_white_max": cell_white_max,
            f"{prefix}_flat_white_frac": flat_white_frac,
        }

    def _cell_white_max(self, gray_patch: np.ndarray) -> float:
        """Max saturated-white ratio over an NxM cell grid (localized decode blowout)."""
        h, w = gray_patch.shape[:2]
        cols, rows = self._cell_grid
        ch, cw = max(1, h // rows), max(1, w // cols)
        best = 0.0
        for i in range(rows):
            for j in range(cols):
                cell = gray_patch[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]

                if cell.size == 0:
                    #print(f"<<<<<<<<<<<<<<< cell ({i}, {j}) is empty >>>>>>>>>>>>>>>>>>>>")
                    continue

                r = float(np.mean(cell >= 1000)) #old 245

                #print(f"<<<<<<<<<<<<<<< cell ({i}, {j}) white ratio: {r} >>>>>>>>>>>>>>>>>>>>")

                if r > best:
                    best = r
        return best

    @staticmethod
    def _flat_white_frac(gray_patch: np.ndarray) -> float:
        """Fraction of the patch that is saturated (>=250) AND flat (near-zero local
        variance) — the signature of a decode block, not a textured highlight."""
        g = gray_patch.astype(np.float32)
        blur = cv2.blur(g, (3, 3))
        local_var = cv2.blur((g - blur) ** 2, (3, 3))
        return float(np.mean((gray_patch >= 250) & (local_var < 5.0)))

    def _artifact_reason(self, metrics: Dict[str, float], prefix: str) -> Optional[str]:
        half_delta = metrics.get(f"{prefix}_half_mean_delta", 0.0)
        max_half = metrics.get(f"{prefix}_max_half_mean", 0.0)
        min_half = metrics.get(f"{prefix}_min_half_mean", 0.0)
        white_ratio = metrics.get(f"{prefix}_white_ratio", 0.0)
        black_ratio = metrics.get(f"{prefix}_black_ratio", 0.0)
        sat_ratio = metrics.get(f"{prefix}_sat_ratio", 0.0)
        sat_std = metrics.get(f"{prefix}_sat_std", 0.0)

        if half_delta >= 85.0 and (max_half >= 220.0 or min_half <= 20.0):
            return f"{prefix}_half_frame_brightness_split"
        if white_ratio >= 0.65:
            return f"{prefix}_large_white_block"
        # Localized decode blowout: a single grid cell fully saturated, or a flat

        # (zero-texture) saturated region — both stay under the global white_ratio.
        if metrics.get(f"{prefix}_cell_white_max", 0.0) >= self._cell_white_ratio:
            return f"{prefix}_cell_white_block"
        
        if metrics.get(f"{prefix}_flat_white_frac", 0.0) >= self._flat_white_fraction:
            return f"{prefix}_flat_white_block"
        if black_ratio >= 0.85:
            return f"{prefix}_large_black_block"
        if sat_ratio >= 0.45 and sat_std >= 45.0:
            return f"{prefix}_abnormal_chroma_artifacts"
        if prefix == "frame" and sat_ratio >= 0.75 and metrics.get("last_good_delta", 0.0) >= 50.0:
            return f"{prefix}_abnormal_chroma_artifacts"
        return None

    def _crop_roi(self, frame: np.ndarray, roi: np.ndarray) -> Optional[np.ndarray]:
        height, width = frame.shape[:2]
        x1 = max(0, int(np.min(roi[:, 0])))
        y1 = max(0, int(np.min(roi[:, 1])))
        x2 = min(width, int(np.max(roi[:, 0])) + 1)
        y2 = min(height, int(np.max(roi[:, 1])) + 1)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]
