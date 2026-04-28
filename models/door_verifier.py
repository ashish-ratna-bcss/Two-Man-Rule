# models/door_verifier.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Optional
import config

class DoorVerifier:
    """Verifies door open/closed state via SSIM comparison."""

    def __init__(self, reference_image_path: str, threshold: float = None, door_roi=None):
        """
        Initialize with reference (closed door) image.

        Args:
            reference_image_path: Path to closed_ref.jpg
            threshold: SSIM threshold (default from config)
        """
        self.reference = cv2.imread(reference_image_path)
        if self.reference is None:
            raise FileNotFoundError(f"Cannot load reference image: {reference_image_path}")

        # Convert to grayscale for SSIM
        self.reference_gray = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY)
        self.threshold = threshold or config.SSIM_THRESHOLD
        self.door_roi = np.array(door_roi, dtype=np.int32) if door_roi is not None else config.DOOR_ROI
        self.last_ssim = None

    def extract_door_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extract DOOR_ROI polygon from frame."""
        if self.door_roi is None:
            raise ValueError("DOOR_ROI not configured")

        # Create mask for polygon
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        points = np.array(self.door_roi, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        if cv2.countNonZero(mask) == 0:
            return frame

        # Extract ROI
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        return roi

    def is_door_open(self, frame: np.ndarray) -> bool:
        """
        Check if door is open by comparing SSIM to reference.

        Returns:
            True if door is OPEN (SSIM < threshold)
            False if door is CLOSED (SSIM >= threshold)
        """
        try:
            reference = self.reference
            if reference.shape[:2] != frame.shape[:2]:
                reference = cv2.resize(reference, (frame.shape[1], frame.shape[0]))

            reference_roi = self.extract_door_roi(reference)
            door_roi = self.extract_door_roi(frame)
            reference_gray = cv2.cvtColor(reference_roi, cv2.COLOR_BGR2GRAY)
            door_roi_gray = cv2.cvtColor(door_roi, cv2.COLOR_BGR2GRAY)

            # Ensure same dimensions
            if door_roi_gray.shape != reference_gray.shape:
                door_roi_gray = cv2.resize(
                    door_roi_gray,
                    (reference_gray.shape[1], reference_gray.shape[0])
                )

            # Calculate SSIM
            result = ssim(
                reference_gray,
                door_roi_gray,
                full=False
            )
            # Handle both tuple (mssim, grad) and scalar return
            self.last_ssim = float(result) if isinstance(result, (int, float, np.floating)) else result[0]

            # Door is CLOSED if SSIM is HIGH
            return self.last_ssim < self.threshold

        except Exception as e:
            print(f"[DoorVerifier] Error: {e}")
            return False

    def get_last_ssim(self) -> Optional[float]:
        """Return last calculated SSIM value."""
        return self.last_ssim
