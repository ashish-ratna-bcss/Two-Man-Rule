# models/door_verifier.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Optional
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
    ):
        reference = cv2.imread(reference_image_path)
        if reference is None:
            raise FileNotFoundError(f"Cannot load reference image: {reference_image_path}")

        self.rx, self.ry, self.rw, self.rh = cv2.boundingRect(door_corner_roi)
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

        self.similarity_threshold = similarity_threshold
        self.intensity_threshold = intensity_threshold if intensity_threshold is not None else 25
        self.debounce_threshold = debounce_threshold
        self.darkening_protection = bool(darkening_protection)

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

        self.candidate_state = False      # False = CLOSED
        self.consecutive_frames_agreed = 0
        self.stable_is_open = False
        self.last_ssim = 1.0
        self._frame_tick = 0

        print(f"[DOOR] Initialized | patch={self.rw}x{self.rh}px | SSIM size: {self.ssim_size} | ref_std={ref_std:.1f}")
        print(
            f"[DOOR] thresholds: similarity={self.similarity_threshold} "
            f"intensity={self.intensity_threshold} motion={self.motion_threshold:.2f}"
            f"{'(adaptive)' if ref_std < 10.0 else ''} debounce_frames={self.debounce_threshold}"
        )

    def verify(self, frame: np.ndarray) -> bool:
        """Returns True if door is OPEN, False if CLOSED with motion gating."""
        try:
            curr_crop = frame[self.ry:self.ry+self.rh, self.rx:self.rx+self.rw]
            curr_patch_raw = cv2.cvtColor(curr_crop, cv2.COLOR_BGR2GRAY)
            curr_patch = cv2.resize(curr_patch_raw, self.ssim_size)

            # 1. Motion Gate (Cheap Pixel-Diff) - Highest Efficiency Gain
            # If the patch hasn't changed at all, skip SSIM entirely
            pixel_diff = cv2.absdiff(curr_patch, self.reference_patch)
            mean_diff = np.mean(pixel_diff)
            
            if mean_diff < self.motion_threshold:
                # No significant motion/lighting change - keep current raw state
                raw_is_open = False # Matches reference (CLOSED)
                self.last_ssim = 1.0
            else:
                # 2. Mean Intensity Check
                curr_mean = np.mean(curr_patch)
                intensity_diff = abs(curr_mean - self.reference_mean)

                # 3. Optimized SSIM (on downscaled patch)
                self.last_ssim = float(ssim(self.reference_patch, curr_patch, full=False))
                self.last_ssim = max(0.0, min(1.0, self.last_ssim))

                texture_open = self.last_ssim < self.similarity_threshold
                # Require SSIM drop in addition to intensity change to avoid false opens
                # Only treat intensity change as an open signal if SSIM also indicates
                # a texture-level change (prevents reflections/lighting spikes from flipping state)
                intensity_open = (
                    intensity_diff > self.intensity_threshold
                    and self.last_ssim < self.similarity_threshold
                )

                if curr_mean < 20.0:  # Blackout protection
                    raw_is_open = False
                else:
                    # Protect against sudden darkening (lights turned off) which
                    # can reduce SSIM but does not mean the door opened. If the
                    # patch is significantly darker than the reference (by at
                    # least the intensity threshold), require a stronger SSIM
                    # drop before treating it as an OPEN event. This behavior
                    # can be disabled via the `darkening_protection` flag.
                    if self.darkening_protection and curr_mean < (self.reference_mean - self.intensity_threshold):
                        # Scene is significantly darker than reference = lights turned off.
                        # SSIM naturally drops against a bright reference regardless of door
                        # state, so it cannot be trusted here. Hold the last stable state so
                        # a lights-off → lights-on cycle never produces a spurious transition.
                        raw_is_open = self.stable_is_open
                    else:
                        raw_is_open = texture_open or intensity_open

            # Debounce Logic
            if raw_is_open == self.candidate_state:
                self.consecutive_frames_agreed += 1
            else:
                self.candidate_state = raw_is_open
                self.consecutive_frames_agreed = 1

            self._frame_tick += 1
            if self._frame_tick % 30 == 0:
                print(f"[DOOR] SSIM: {self.last_ssim:.3f} | Diff: {mean_diff:.1f} | "
                      f"Stable: {'OPEN' if self.stable_is_open else 'CLOSED'}")

            if self.consecutive_frames_agreed >= self.debounce_threshold:
                if self.stable_is_open != self.candidate_state:
                    print(f"[DOOR] *** STATE CHANGE: {'OPEN' if self.candidate_state else 'CLOSED'} ***")
                    self.stable_is_open = self.candidate_state
                self.consecutive_frames_agreed = self.debounce_threshold

            return self.stable_is_open

        except Exception as e:
            print(f"[DoorVerifier] Error: {e}")
            return self.stable_is_open

    def get_last_ssim(self) -> Optional[float]:
        return self.last_ssim

    def is_transition_pending(self) -> bool:
        return self.candidate_state != self.stable_is_open
