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
        similarity_threshold: float = 0.75,
        debounce_threshold: int = 8
    ):
        # Load reference image
        reference = cv2.imread(reference_image_path)
        if reference is None:
            raise FileNotFoundError(f"Cannot load reference image: {reference_image_path}")

        # Parse DOOR_CORNER_ROI into crop coords using bounding rect
        # config.DOOR_CORNER_ROI is a numpy polygon array
        self.rx, self.ry, self.rw, self.rh = cv2.boundingRect(config.DOOR_CORNER_ROI)

        # Extract and store reference patch (grayscale)
        ref_crop = reference[self.ry:self.ry+self.rh, self.rx:self.rx+self.rw]
        self.reference_patch = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)

        # Thresholds
        self.similarity_threshold = similarity_threshold
        self.debounce_threshold = debounce_threshold

        # Debounce state
        self.candidate_state = False      # False = CLOSED
        self.consecutive_frames_agreed = 0
        self.stable_is_open = False

        # Debug
        self.last_ssim = None
        self._frame_tick = 0

        print(f"[DOOR] Initialized | Corner ROI: {config.DOOR_CORNER_ROI} | Patch shape: {self.reference_patch.shape}")

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

            # SSIM comparison
            self.last_ssim = float(ssim(self.reference_patch, curr_patch, full=False))

            # LOW similarity = corner gone = door OPEN
            raw_is_open = self.last_ssim < self.similarity_threshold

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
                    f"[DOOR] SSIM: {self.last_ssim:.3f} | "
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
                        f"(SSIM={self.last_ssim:.3f}) ***"
                    )
                    self.stable_is_open = self.candidate_state
                # Cap to prevent overflow
                self.consecutive_frames_agreed = self.debounce_threshold

            return self.stable_is_open

        except Exception as e:
            print(f"[DoorVerifier] Error in verify(): {e}")
            return self.stable_is_open

    def get_last_ssim(self) -> Optional[float]:
        return self.last_ssim