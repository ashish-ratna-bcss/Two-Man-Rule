# io/video_handler.py
import cv2
import time
import os
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
import config

IST = timezone(timedelta(hours=5, minutes=30))


class VideoHandler:
    """
    Synchronous, queue-free video input.

    Every read_frame() call performs cap.read() directly on the consumer
    thread and pins the IST wall-clock immediately after the grab. No
    background grabber thread, no 1-slot frame buffer, no condition wait,
    no dropped-frame accounting. This eliminates queue lag and stale
    frames between RTSP decode and inference. The frame returned is
    always the freshest one FFmpeg has demuxed at the moment of the call.

    RTSP keeps CAP_PROP_BUFFERSIZE=1 plus FFmpeg low_delay/nobuffer flags
    so the OS-side socket buffer stays drained.
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

        # Latest pinned IST timestamp (set in read_frame).
        self.frame_ist: Optional[datetime] = None
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

    def read_frame(
        self,
        block: bool = False,
        timeout: Optional[float] = None,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[datetime]]:
        """
        Direct synchronous read. Returns (ret, frame, frame_ist).

        frame_ist is wall-clock pinned IMMEDIATELY after a successful
        cap.read(). No queue between decode and timestamp. block/timeout
        kwargs are retained for API compatibility; they no longer gate
        anything because the read is synchronous.
        """
        if not self.running:
            return False, None, None

        # Pace file playback to real fps so timing logic mirrors live behavior.
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
                return False, None, None
            ret, frame, grab_ist = self._reconnect_rtsp()
            if not ret:
                self.running = False
                self.ret = False
                return False, None, None

        self.ret = True
        self.current_frame_idx += 1
        self.last_frame_time = time.time()
        self.frame_ist = grab_ist
        return True, frame, grab_ist

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
        try:
            self.cap.release()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
