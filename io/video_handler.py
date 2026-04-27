# io/video_handler.py
import cv2
from typing import Optional, Tuple

class VideoHandler:
    """Video input/output handling."""

    def __init__(self, video_source: str):
        """
        Initialize video handler.

        Args:
            video_source: Path to video file or 0 for webcam
        """
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {video_source}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0

    def read_frame(self) -> Tuple[bool, Optional]:
        """Read next frame."""
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx += 1
        return ret, frame

    def get_fps(self) -> float:
        """Get video FPS."""
        return self.fps

    def get_dimensions(self) -> Tuple[int, int]:
        """Get frame dimensions (width, height)."""
        return (self.width, self.height)

    def get_total_frames(self) -> int:
        """Get total frame count."""
        return self.total_frames

    def get_progress(self) -> float:
        """Get progress as percentage (0-100)."""
        if self.total_frames == 0:
            return 0
        return (self.current_frame_idx / self.total_frames) * 100

    def release(self):
        """Release video capture."""
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
