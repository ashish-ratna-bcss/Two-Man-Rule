# io/video_handler.py
import cv2
import time
import threading
import os
import numpy as np
from typing import Optional, Tuple
import config

class VideoHandler:
    """
    Production-grade Video input handling with threaded capture and condition-based waiting.
    Prevents RTSP latency accumulation and CPU spinning.
    """

    def __init__(self, video_source, reconnect_delay: float = 5.0, max_reconnect_attempts: int = 0):
        self.video_source = video_source
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self._is_rtsp = isinstance(video_source, str) and video_source.lower().startswith("rtsp://")
        
        # Telemetry
        self.dropped_frames = 0
        self.reconnect_count = 0
        self.last_frame_time = 0
        self.queue_delay = 0.0
        
        self._preserve_file_frames = (
            not self._is_rtsp and getattr(config, "PRESERVE_FILE_FRAMES", True)
        )

        # Threading state
        self.frame = None
        self.ret = False
        self.running = True
        self.condition = threading.Condition() # Using Condition for efficient waiting
        
        # Low-latency FFMPEG configuration
        if self._is_rtsp and config.RTSP_LOW_LATENCY:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|nobuffer;|low_delay;"

        self.cap = self._open()
        
        # Metadata
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0

        # File replays are consumed synchronously so every frame is processed.
        # Live RTSP stays threaded/latest-frame to prevent latency buildup.
        self.thread = None
        if not self._preserve_file_frames:
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

    def _open(self) -> cv2.VideoCapture:
        """Open video source with optimized buffer settings."""
        cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {self.video_source}")
        
        if self._is_rtsp:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        return cap

    def _capture_loop(self):
        """Background thread to continuously grab the latest frame."""
        _next_frame_time = time.perf_counter()
        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                if not self._is_rtsp:
                    with self.condition:
                        self.ret = False
                        self.running = False
                        self.condition.notify_all()
                    break

                self.reconnect_count += 1
                attempts = 0
                while self.running and (self.max_reconnect_attempts == 0 or attempts < self.max_reconnect_attempts):
                    attempts += 1
                    print(f"[VIDEO] RTSP lost. Reconnect {attempts} in {self.reconnect_delay}s...")
                    time.sleep(self.reconnect_delay)
                    self.cap.release()
                    try:
                        self.cap = self._open()
                        ret, frame = self.cap.read()
                        if ret:
                            print(f"[VIDEO] RTSP reconnected.")
                            break
                    except Exception as e:
                        print(f"[VIDEO] Reconnect failed: {e}")

                if not ret:
                    with self.condition:
                        self.ret = False
                        self.running = False
                        self.condition.notify_all()
                    break

            # Threaded file mode is only used when PRESERVE_FILE_FRAMES=False.
            # RTSP has no throttle and normally delivers latest-frame only.
            if not self._is_rtsp:
                now = time.perf_counter()
                sleep_time = _next_frame_time - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                frame_interval = 1.0 / max(self.fps, 1.0)
                _next_frame_time = max(time.perf_counter(), _next_frame_time) + frame_interval

            with self.condition:
                if self.frame is not None:
                    if self._is_rtsp and not getattr(config, "RTSP_PREFER_REALTIME", True):
                        while self.running and self.frame is not None:
                            self.condition.wait(timeout=0.05)
                        if not self.running:
                            break
                    elif self.frame is not None:
                        self.dropped_frames += 1
                self.frame = frame
                self.ret = True
                self.last_frame_time = time.time()
                self.current_frame_idx += 1
                self.condition.notify_all() # Wake up read_frame()

    def read_frame(self, block: bool = False, timeout: Optional[float] = None) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the latest frame.
        If block=True, waits until a new frame is available.
        """
        if self._preserve_file_frames:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                self.ret = False
                return False, None

            self.ret = True
            self.current_frame_idx += 1
            self.last_frame_time = time.time()
            self.queue_delay = 0.0
            return True, frame

        with self.condition:
            if block:
                # Wait for a frame to be available
                while self.running and self.frame is None:
                    if not self.condition.wait(timeout=timeout):
                        return True, None # Timeout reached
            
            if not self.running:
                return False, None
            
            if not self.ret:
                return False, None
            
            frame = self.frame
            self.frame = None
            
            if self.last_frame_time > 0:
                self.queue_delay = time.time() - self.last_frame_time
                
            return True, frame

    def get_fps(self) -> float:
        return self.fps

    def get_dimensions(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def get_total_frames(self) -> int:
        return self.total_frames

    def get_progress(self) -> float:
        if self.total_frames <= 0: return 0
        return (self.current_frame_idx / self.total_frames) * 100

    def get_telemetry(self) -> dict:
        return {
            "dropped_frames": self.dropped_frames,
            "reconnect_count": self.reconnect_count,
            "queue_delay_ms": self.queue_delay * 1000.0,
            "frame_idx": self.current_frame_idx
        }

    def release(self):
        self.running = False
        with self.condition:
            self.condition.notify_all()
        if hasattr(self, 'thread'):
            if self.thread is not None:
                self.thread.join(timeout=2.0)
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
