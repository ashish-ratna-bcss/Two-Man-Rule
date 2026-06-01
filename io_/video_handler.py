import cv2
import time
import os
import numpy as np
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
import config

IST = timezone(timedelta(hours=5, minutes=30))


class VideoHandler:
    """
    Asynchronous, non-blocking video input via background thread.

    A dedicated background thread continuously pulls frames from the RTSP
    stream as fast as possible. It stores only the absolute most recent
    frame in memory, overwriting any stale frames. This guarantees that:
    1. The OS-level network socket buffer stays completely drained, preventing
       I-frame UDP packet drops and "gray frame" corruptions.
    2. Any call to read_frame() instantly returns the freshest possible
       frame without any stale backlog lag.
    """

    def __init__(
        self,
        video_source,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 0,
    ):
        self.video_source = video_source
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self._is_rtsp = isinstance(video_source, str) and video_source.lower().startswith("rtsp://")

        # Telemetry
        self.reconnect_count = 0
        self.last_frame_time = 0.0

        # Threading mechanisms
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_frame_ist = None
        self._new_frame_event = threading.Event()
        
        self.ret = False
        self.running = True

        # File replay pacing (so synchronous file reads track real fps).
        self._file_pace = (
            not self._is_rtsp and getattr(config, "PRESERVE_FILE_FRAMES", True) is False
        )
        self._next_file_frame_time: Optional[float] = None

        # Low-latency FFMPEG configuration.
        if self._is_rtsp and config.RTSP_LOW_LATENCY:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|nobuffer;|low_delay;"

        self.cap = self._open()

        # Metadata
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0

        # Start the background grabber thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {self.video_source}")
        if self._is_rtsp:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reconnect_rtsp(self) -> Tuple[bool, Optional[np.ndarray], Optional[datetime]]:
        self.reconnect_count += 1
        attempts = 0
        while self.running and (
            self.max_reconnect_attempts == 0 or attempts < self.max_reconnect_attempts
        ):
            attempts += 1
            print(f"[VIDEO] RTSP lost. Reconnect {attempts} in {self.reconnect_delay}s...")
            time.sleep(self.reconnect_delay)
            try:
                self.cap.release()
            except Exception:
                pass
            try:
                self.cap = self._open()
                ret, frame = self.cap.read()
                if ret:
                    print("[VIDEO] RTSP reconnected.")
                    return True, frame, datetime.now(IST)
            except Exception as e:
                print(f"[VIDEO] Reconnect failed: {e}")
        return False, None, None

    def _update(self):
        """Background thread loop to continuously drain frames."""
        while self.running:
            if self._file_pace:
                now_pc = time.perf_counter()
                if self._next_file_frame_time is None:
                    self._next_file_frame_time = now_pc
                sleep_time = self._next_file_frame_time - now_pc
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self._next_file_frame_time = (
                    max(time.perf_counter(), self._next_file_frame_time)
                    + 1.0 / max(self.fps, 1.0)
                )

            ret, frame = self.cap.read()
            grab_ist = datetime.now(IST) if ret else None

            if not ret:
                if not self._is_rtsp:
                    self.running = False
                    self.ret = False
                    self._new_frame_event.set()
                    break
                ret, frame, grab_ist = self._reconnect_rtsp()
                if not ret:
                    self.running = False
                    self.ret = False
                    self._new_frame_event.set()
                    break

            with self._lock:
                self._latest_frame = frame
                self._latest_frame_ist = grab_ist
                self.ret = True
            self._new_frame_event.set()

    def read_frame(
        self,
        block: bool = False,
        timeout: Optional[float] = None,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[datetime]]:
        """
        Fetch the absolute latest frame demuxed by the background thread.
        """
        if not self.running:
            return False, None, None

        if block:
            got_new = self._new_frame_event.wait(timeout)
            if not got_new:
                # Timeout reached without a new frame; return True (stream still alive), but None for frame
                return self.ret, None, None

        with self._lock:
            frame = self._latest_frame
            grab_ist = self._latest_frame_ist
            # Require the background thread to fetch a fresh frame for the next call
            self._new_frame_event.clear()

        if frame is not None:
            self.current_frame_idx += 1
            self.last_frame_time = time.time()
            self.frame_ist = grab_ist
            return True, frame, grab_ist
        else:
            return self.ret, None, None

    def get_fps(self) -> float:
        return self.fps

    def get_dimensions(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def get_total_frames(self) -> int:
        return self.total_frames

    def get_progress(self) -> float:
        if self.total_frames <= 0:
            return 0
        return (self.current_frame_idx / self.total_frames) * 100

    def get_telemetry(self) -> dict:
        return {
            "reconnect_count": self.reconnect_count,
            "frame_idx": self.current_frame_idx,
        }

    def release(self):
        self.running = False
        self._new_frame_event.set() # Unblock if waiting
        try:
            self.cap.release()
        except Exception:
            pass
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
