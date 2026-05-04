# models/door_verifier.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Optional
import config

class DoorVerifier:

    def __init__(
        self,
        reference_image_path: str,
        door_corner_roi: np.ndarray,
        similarity_threshold: float = 0.75,
        debounce_threshold: int = 10
    ):
        # Load reference image
        reference = cv2.imread(reference_image_path)
        if reference is None:
            raise FileNotFoundError(f"Cannot load reference image: {reference_image_path}")

        # Parse door_corner_roi into crop coords using bounding rect
        # door_corner_roi is a numpy polygon array
        self.rx, self.ry, self.rw, self.rh = cv2.boundingRect(door_corner_roi)

        # Extract and store reference patch (grayscale)
        ref_crop = reference[self.ry:self.ry+self.rh, self.rx:self.rx+self.rw]
        self.reference_patch = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
        self.reference_mean = np.mean(self.reference_patch)

        # Thresholds
        self.similarity_threshold = similarity_threshold
        self.intensity_threshold = 25  # Meaningful brightness shift
        self.debounce_threshold = debounce_threshold

        # Debounce state
        self.candidate_state = False      # False = CLOSED
        self.consecutive_frames_agreed = 0
        self.stable_is_open = False

        # Debug
        self.last_ssim = None
        self._frame_tick = 0

        print(f"[DOOR] Initialized | Corner ROI: {door_corner_roi.tolist()} | Patch shape: {self.reference_patch.shape}")

    def verify(self, frame: np.ndarray) -> bool:
        """
        Returns True if door is OPEN, False if CLOSED.
        """
        try:
            # Crop corner ROI from current frame
            curr_crop = frame[self.ry:self.ry+self.rh, self.rx:self.rx+self.rw]
            curr_patch = cv2.cvtColor(curr_crop, cv2.COLOR_BGR2GRAY)

            # Resize if shape mismatch (camera resolution change etc.)
            if curr_patch.shape != self.reference_patch.shape:
                curr_patch = cv2.resize(
                    curr_patch,
                    (self.reference_patch.shape[1], self.reference_patch.shape[0])
                )

            # Mean intensity check
            curr_mean = np.mean(curr_patch)
            intensity_diff = abs(curr_mean - self.reference_mean)

            # SSIM comparison
            self.last_ssim = float(ssim(self.reference_patch, curr_patch, full=False))
            self.last_ssim = max(0.0, min(1.0, self.last_ssim))

            # Door is OPEN if:
            # 1. Texture has deviated (SSIM low)
            # 2. OR Brightness has shifted significantly (even if texture looks similar)
            texture_open = self.last_ssim < self.similarity_threshold
            intensity_open = intensity_diff > self.intensity_threshold
            
            # LIGHT CHECK: If it is extremely dark (lights off), do NOT trigger false open
            if curr_mean < 20.0:
                raw_is_open = False
            else:
                raw_is_open = texture_open or intensity_open

            # Debounce
            if raw_is_open == self.candidate_state:
                self.consecutive_frames_agreed += 1
            else:
                self.candidate_state = raw_is_open
                self.consecutive_frames_agreed = 1

            # Debug every 30 frames
            self._frame_tick += 1
            if self._frame_tick % 30 == 0:
                print(
                    f"[DOOR] SSIM: {self.last_ssim:.3f} | Mean Diff: {intensity_diff:.1f} | "
                    f"Stable: {'OPEN' if self.stable_is_open else 'CLOSED'} | "
                    f"Debounce: {self.consecutive_frames_agreed}/{self.debounce_threshold} "
                    f"-> {'OPEN' if self.candidate_state else 'CLOSED'}"
                )

            # Flip stable state only after debounce threshold met
            if self.consecutive_frames_agreed >= self.debounce_threshold:
                if self.stable_is_open != self.candidate_state:
                    print(
                        f"[DOOR] *** State Flip: "
                        f"{'CLOSED -> OPEN' if self.candidate_state else 'OPEN -> CLOSED'} "
                        f"(SSIM={self.last_ssim:.3f}, IntDiff={intensity_diff:.1f}) ***"
                    )
                    self.stable_is_open = self.candidate_state
                self.consecutive_frames_agreed = self.debounce_threshold

            return self.stable_is_open

        except Exception as e:
            print(f"[DoorVerifier] Error in verify(): {e}")
            return self.stable_is_open

    def get_last_ssim(self) -> Optional[float]:
        return self.last_ssim

    def is_transition_pending(self) -> bool:
        return self.candidate_state != self.stable_is_open
