# models/door_verifier.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Optional
from datetime import datetime
import config

class DoorVerifier:
    """
    Optimized door state verification using motion gating and downscaled SSIM.
    """

    def __init__(
        self,
        reference_image_path: str,
        door_corner_roi: np.ndarray,
        similarity_threshold: float = 0.75,
        debounce_threshold: int = 20,
        intensity_threshold: float = None,
        motion_threshold: float = None,
        darkening_protection: bool = True,
        min_visible_ratio: float = None,
    ):
        reference = cv2.imread(reference_image_path)
        if reference is None:
            raise FileNotFoundError(f"Cannot load reference image: {reference_image_path}")

        self.rx, self.ry, self.rw, self.rh = cv2.boundingRect(door_corner_roi)
        self._door_corner_polygon = door_corner_roi.reshape(-1, 2).astype(np.int32)
        ref_crop = reference[self.ry:self.ry+self.rh, self.rx:self.rx+self.rw]
        raw_patch = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)

        # Scale to fixed 100px on the LONGER dimension to preserve aspect ratio.
        # Avoids extreme squishing (e.g. 206x65 → 100x31) and extreme upscaling
        # (e.g. 22x43 → 100x195) that both degrade SSIM reliability.
        if self.rw >= self.rh:
            new_w = 100
            new_h = max(16, int(100 * self.rh / max(self.rw, 1)))
        else:
            new_h = 100
            new_w = max(16, int(100 * self.rw / max(self.rh, 1)))
        self.ssim_size = (new_w, new_h)
        self.reference_patch = cv2.resize(raw_patch, self.ssim_size)
        self.reference_mean = np.mean(self.reference_patch)
        ref_std = float(np.std(self.reference_patch))

        self._roi_mask = self._build_roi_mask()
        self._roi_area_pixels = int(np.count_nonzero(self._roi_mask))
        self.last_visible_ratio = 1.0

        self.similarity_threshold = similarity_threshold
        self.intensity_threshold = intensity_threshold if intensity_threshold is not None else 25
        self.debounce_threshold = debounce_threshold
        self.darkening_protection = bool(darkening_protection)
        self.min_visible_ratio = (
            float(min_visible_ratio)
            if min_visible_ratio is not None
            else float(config.DOOR_CORNER_MIN_VISIBLE_RATIO)
        )

        # Motion gate: for very low-texture patches (flat door surface) the
        # configured threshold can be higher than the actual per-pixel change a
        # real door opening produces.  Cap it at 60 % of the patch's own std so
        # the gate is always proportional to the patch's inherent variation.
        raw_motion_thresh = motion_threshold if motion_threshold is not None else 3.0
        if ref_std < 10.0:
            adaptive_cap = max(1.5, ref_std * 0.6)
            self.motion_threshold = min(raw_motion_thresh, adaptive_cap)
        else:
            self.motion_threshold = raw_motion_thresh

        self.debounce_seconds = float(debounce_threshold) / 25.0
        self.candidate_state = False      # False = CLOSED
        self.candidate_state_start_time = None
        self.stable_is_open = False
        self.has_stabilized = False
        self.last_ssim = 1.0
        self.last_curr_mean = None
        self.last_intensity_diff = None
        self.last_mean_diff = None
        self._frame_tick = 0

        print(f"[DOOR] Initialized | patch={self.rw}x{self.rh}px | SSIM size: {self.ssim_size} | ref_std={ref_std:.1f}")
        print(
            f"[DOOR] thresholds: similarity={self.similarity_threshold} "
            f"intensity={self.intensity_threshold} motion={self.motion_threshold:.2f}"
            f"{'(adaptive)' if ref_std < 10.0 else ''} debounce_frames={self.debounce_threshold}"
        )

    def _build_roi_mask(self) -> np.ndarray:
        mask = np.zeros((self.ssim_size[1], self.ssim_size[0]), dtype=np.uint8)
        local_polygon = self._door_corner_polygon.astype(np.float32)
        local_polygon[:, 0] = (local_polygon[:, 0] - self.rx) * self.ssim_size[0] / max(self.rw, 1)
        local_polygon[:, 1] = (local_polygon[:, 1] - self.ry) * self.ssim_size[1] / max(self.rh, 1)
        local_polygon = np.rint(local_polygon).astype(np.int32)
        cv2.fillPoly(mask, [local_polygon], 1)
        return mask

    def _build_visible_mask(self, tracked_persons: Optional[Dict[int, Dict]]) -> np.ndarray:
        occlusion_mask = np.zeros_like(self._roi_mask, dtype=np.uint8)
        if tracked_persons:
            scale_x = self.ssim_size[0] / max(self.rw, 1)
            scale_y = self.ssim_size[1] / max(self.rh, 1)
            for person in tracked_persons.values():
                bbox = person.get("bbox")
                if bbox is None or len(bbox) < 4:
                    continue
                x1 = max(float(bbox[0]), float(self.rx))
                y1 = max(float(bbox[1]), float(self.ry))
                x2 = min(float(bbox[2]), float(self.rx + self.rw))
                y2 = min(float(bbox[3]), float(self.ry + self.rh))
                if x2 <= x1 or y2 <= y1:
                    continue

                lx1 = int(np.floor((x1 - self.rx) * scale_x))
                ly1 = int(np.floor((y1 - self.ry) * scale_y))
                lx2 = int(np.ceil((x2 - self.rx) * scale_x))
                ly2 = int(np.ceil((y2 - self.ry) * scale_y))

                lx1 = max(0, min(lx1, self.ssim_size[0] - 1))
                ly1 = max(0, min(ly1, self.ssim_size[1] - 1))
                lx2 = max(0, min(lx2, self.ssim_size[0]))
                ly2 = max(0, min(ly2, self.ssim_size[1]))
                if lx2 <= lx1 or ly2 <= ly1:
                    continue

                cv2.rectangle(occlusion_mask, (lx1, ly1), (lx2 - 1, ly2 - 1), 1, thickness=-1)

        visible_mask = np.where((self._roi_mask == 1) & (occlusion_mask == 0), 1, 0).astype(np.uint8)
        visible_pixels = int(np.count_nonzero(visible_mask))
        self.last_visible_ratio = (visible_pixels / self._roi_area_pixels) if self._roi_area_pixels else 0.0
        return visible_mask

    def _run_verification(self, curr_patch: np.ndarray, reference_patch: np.ndarray, visible_mask: np.ndarray, ts_ist: datetime) -> bool:
        if np.count_nonzero(visible_mask) == 0:
            return self.stable_is_open

        # Analyze brightness and motion ONLY on the non-occluded pixels
        visible_curr = curr_patch[visible_mask == 1]
        visible_ref = reference_patch[visible_mask == 1]

        self.reference_mean = float(np.mean(visible_ref))
        curr_mean = float(np.mean(visible_curr))

        pixel_diff = cv2.absdiff(visible_curr, visible_ref)
        mean_diff = float(np.mean(pixel_diff))
        
        self.last_mean_diff = mean_diff
        self.last_curr_mean = curr_mean
        self.last_intensity_diff = abs(curr_mean - self.reference_mean)

        active_threshold = self.similarity_threshold
        if mean_diff < self.motion_threshold:
            raw_is_open = False
            self.last_ssim = 1.0
        else:
            # Normalize brightness based strictly on the visible portion of the door
            mean_offset = self.reference_mean - curr_mean
            adjusted_curr = np.clip(curr_patch.astype(np.int16) + mean_offset, 0, 255).astype(np.uint8)

            # Build composite patch for SSIM: 
            # Paste the pristine reference pixels over occluded areas, 
            # and use the brightness-adjusted current pixels for visible areas.
            composite_patch = reference_patch.copy()
            composite_patch[visible_mask == 1] = adjusted_curr[visible_mask == 1]

            self.last_ssim = float(ssim(reference_patch, composite_patch, full=False, data_range=255))
            self.last_ssim = max(0.0, min(1.0, self.last_ssim))

            # Twilight Protection Layer:
            # In dim ambient light (e.g., early morning before lights are on), 
            # the camera loses physical contrast, artificially lowering the SSIM score.
            # We dynamically relax the threshold here to prevent false "OPEN" triggers,
            # while still allowing massive structural changes (actual openings) to be caught.
            if self.darkening_protection and 20.0 <= curr_mean < 45.0:
                active_threshold = self.similarity_threshold * 0.85

            ssim_changed = self.last_ssim < active_threshold

            if curr_mean < 20.0 and self.darkening_protection:
                raw_is_open = False
            else:
                raw_is_open = ssim_changed

        if raw_is_open == self.candidate_state:
            if self.candidate_state_start_time is None:
                self.candidate_state_start_time = ts_ist
        else:
            self.candidate_state = raw_is_open
            self.candidate_state_start_time = ts_ist

        if raw_is_open or self.candidate_state or self.stable_is_open:
            print(f"[DEBUG DOOR] tick={self._frame_tick} raw={raw_is_open} cand={self.candidate_state} "
                  f"SSIM={self.last_ssim:.3f} (thresh={active_threshold:.3f}) "
                  f"Diff={self.last_mean_diff:.1f} (thresh={self.motion_threshold:.1f}) "
                  f"VisibleRatio={self.last_visible_ratio:.2f}")

        self._frame_tick += 1
        if self._frame_tick % 30 == 0:
            print(f"[DOOR] SSIM: {self.last_ssim:.3f} | Diff: {self.last_mean_diff:.1f} | "
                f"Intensity: {self.last_curr_mean:.1f} (Δ{self.last_intensity_diff:.1f}) | "
                f"Stable: {'OPEN' if self.stable_is_open else 'CLOSED'}")

        # Compute elapsed time in candidate state
        if self.candidate_state_start_time is not None:
            elapsed_seconds = (ts_ist - self.candidate_state_start_time).total_seconds()
        else:
            elapsed_seconds = 0.0

        if elapsed_seconds >= self.debounce_seconds:
            if not self.has_stabilized:
                self.stable_is_open = self.candidate_state
                self.has_stabilized = True
                print(f"[DOOR] Initial stabilization: {'OPEN' if self.stable_is_open else 'CLOSED'}")
            elif self.stable_is_open != self.candidate_state:
                print(f"[DOOR] *** STATE CHANGE: {'OPEN' if self.candidate_state else 'CLOSED'} ***")
                self.stable_is_open = self.candidate_state

        return self.stable_is_open

    def verify(self, frame: np.ndarray, tracked_persons: Optional[Dict[int, Dict]] = None, ts_ist: Optional[datetime] = None) -> bool:
        """Returns True if door is OPEN, False if CLOSED with motion gating."""
        try:
            curr_crop = frame[self.ry:self.ry+self.rh, self.rx:self.rx+self.rw]
            curr_patch_raw = cv2.cvtColor(curr_crop, cv2.COLOR_BGR2GRAY)
            curr_patch = cv2.resize(curr_patch_raw, self.ssim_size)

            visible_mask = self._build_visible_mask(tracked_persons)
            if self.last_visible_ratio < self.min_visible_ratio:
                return self.stable_is_open

            if ts_ist is None:
                from datetime import timezone, timedelta
                ts_ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))

            return self._run_verification(curr_patch, self.reference_patch, visible_mask, ts_ist)

        except Exception as e:
            print(f"[DoorVerifier] Error: {e}")
            return self.stable_is_open

    def get_last_ssim(self) -> Optional[float]:
        return self.last_ssim

    def get_last_intensity(self) -> Optional[float]:
        return self.last_curr_mean

    def get_last_intensity_diff(self) -> Optional[float]:
        return self.last_intensity_diff

    def get_last_mean_diff(self) -> Optional[float]:
        return self.last_mean_diff

    def is_transition_pending(self) -> bool:
        return self.candidate_state != self.stable_is_open
