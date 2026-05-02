# io/video_handler.py
import cv2
import time
import numpy as np
from typing import Optional, Tuple

class VideoHandler:
    """Video input/output handling."""

    def __init__(self, video_source, reconnect_delay: float = 5.0, max_reconnect_attempts: int = 0):
        """
        Args:
            video_source: RTSP URL, file path, or webcam index.
            reconnect_delay: Seconds to wait between reconnect attempts (RTSP only).
            max_reconnect_attempts: 0 = retry forever (daemon mode).
        """
        self.video_source = video_source
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self._is_rtsp = isinstance(video_source, str) and video_source.lower().startswith("rtsp://")
        self.cap = self._open()
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.video_source}")
        return cap

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame. For RTSP sources, auto-reconnects on drop."""
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx += 1
            return ret, frame

        if not self._is_rtsp:
            return False, None

        # RTSP dropped — attempt reconnect
        attempts = 0
        while self.max_reconnect_attempts == 0 or attempts < self.max_reconnect_attempts:
            attempts += 1
            print(f"[VIDEO] RTSP stream lost. Reconnect attempt {attempts} in {self.reconnect_delay}s...")
            time.sleep(self.reconnect_delay)
            self.cap.release()
            try:
                self.cap = self._open()
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame_idx += 1
                    print(f"[VIDEO] RTSP reconnected after {attempts} attempt(s).")
                    return ret, frame
            except RuntimeError as e:
                print(f"[VIDEO] Reconnect failed: {e}")

        print("[VIDEO] Max reconnect attempts reached. Giving up.")
        return False, None

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
