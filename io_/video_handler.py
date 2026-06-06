import cv2
import queue
import time
import numpy as np
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
import config

IST = timezone(timedelta(hours=5, minutes=30))

_RTSP_TRANSPORT_MAP = {
    "tcp": "tcp",
    "udp": "udp",
    "udp-mcast": "udp_multicast",
    "http": "http",
    "tls": "tls",
}


class _OpenCVCapture:
    def __init__(self, video_source):
        self.cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(video_source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {video_source}")

    def read(self):
        return self.cap.read()

    def get(self, prop_id):
        return self.cap.get(prop_id)

    def release(self):
        self.cap.release()


class _PyAVRTSPCapture:
    """
    RTSP capture via PyAV (FFmpeg bindings). Server-safe: pure Python wheel,
    no system GStreamer or gobject-introspection required.

    Background thread decodes into a queue(maxsize=1), dropping stale frames
    so the caller always gets the newest available frame.
    """

    def __init__(
        self,
        rtsp_url: str,
        *,
        transport: str,
        latency_ms: int,
        drop_on_latency: bool,
        read_timeout_seconds: float,
        startup_timeout_seconds: float,
    ):
        try:
            import av as _av
            self._av = _av
        except ImportError as e:
            raise RuntimeError(
                "RTSP ingest requires PyAV. Install with: pip install av>=12.0.0"
            ) from e

        self.rtsp_url = rtsp_url
        self.read_timeout_seconds = max(0.1, float(read_timeout_seconds))
        self._fps = 0.0
        self._width = 0
        self._height = 0
        self._error: Optional[str] = None
        self._pending_frame: Optional[np.ndarray] = None
        self._running = True

        av_transport = _RTSP_TRANSPORT_MAP.get(str(transport).lower(), "tcp")
        options = {
            "rtsp_transport": av_transport,
            "buffer_size": "4096000",
            "max_delay": str(max(0, int(latency_ms)) * 1000),  # microseconds
            "stimeout": str(int(startup_timeout_seconds * 1_000_000)),
            # Drop corrupt/partial frames at the FFmpeg layer so torn/blocky frames
            # from RTSP packet loss never reach the decoder output — the downstream
            # frame-quality guard then sees far fewer bad frames per stream.
            "fflags": "discardcorrupt",
            "err_detect": "crccheck",
        }

        try:
            self._container = self._av.open(rtsp_url, options=options)
        except Exception as e:
            raise RuntimeError(f"PyAV failed to open RTSP stream: {e}") from e

        video_streams = [s for s in self._container.streams if s.type == "video"]
        if not video_streams:
            raise RuntimeError("PyAV: no video stream found in RTSP source.")
        self._video_stream = video_streams[0]
        self._fps = float(self._video_stream.average_rate or config.DEFAULT_FPS)
        self._width = self._video_stream.width or 0
        self._height = self._video_stream.height or 0

        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._decode_loop, daemon=True)
        self._thread.start()

        self._prime_first_frame(startup_timeout_seconds)

    def _decode_loop(self):
        try:
            for frame in self._container.decode(self._video_stream):
                if not self._running:
                    return
                if self._width == 0:
                    self._width = frame.width
                    self._height = frame.height
                bgr = frame.to_ndarray(format="bgr24")
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                self._queue.put(bgr)
        except Exception as e:
            if self._running:
                self._error = str(e)
                self._queue.put(None)  # sentinel: error / EOS

    def _prime_first_frame(self, startup_timeout_seconds: float):
        try:
            frame = self._queue.get(timeout=startup_timeout_seconds)
        except queue.Empty:
            raise RuntimeError(
                f"PyAV RTSP pipeline did not deliver a frame within {startup_timeout_seconds:.1f}s."
            )
        if frame is None:
            raise RuntimeError(
                "PyAV RTSP pipeline error during startup."
                + (f" {self._error}" if self._error else "")
            )
        self._pending_frame = frame

    def read(self, timeout_seconds: Optional[float] = None):
        if self._pending_frame is not None:
            frame = self._pending_frame
            self._pending_frame = None
            return True, frame

        timeout = self.read_timeout_seconds if timeout_seconds is None else max(0.1, float(timeout_seconds))
        try:
            frame = self._queue.get(timeout=timeout)
            if frame is None:
                return False, None  # error or EOS
            return True, frame
        except queue.Empty:
            return True, None  # no sample yet — caller handles as transient timeout

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return self._fps or config.DEFAULT_FPS
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self._width
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._height
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return 0
        return 0

    def release(self):
        self._running = False
        try:
            self._container.close()
        except Exception:
            pass


class VideoHandler:
    """
    Asynchronous video input. RTSP sources decoded via PyAV (FFmpeg);
    local files via OpenCV.
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

        self.reconnect_count = 0
        self.read_timeout_count = 0
        self.last_frame_time = 0.0

        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_frame_ist = None
        self._new_frame_event = threading.Event()

        self.ret = False
        self.running = True

        self._file_pace = (
            not self._is_rtsp and getattr(config, "PRESERVE_FILE_FRAMES", True) is False
        )
        self._next_file_frame_time: Optional[float] = None

        self.cap = self._open()

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0

        if self._is_rtsp:
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _open(self):
        if self._is_rtsp:
            backend = str(getattr(config, "RTSP_INGEST_BACKEND", "pyav")).lower()
            if backend != "pyav":
                raise RuntimeError(
                    f"Unsupported RTSP_INGEST_BACKEND={backend!r}. Only 'pyav' is supported."
                )
            return _PyAVRTSPCapture(
                self.video_source,
                transport=getattr(config, "RTSP_TRANSPORT", "tcp"),
                latency_ms=int(getattr(config, "RTSP_JITTER_LATENCY_MS", 1000)),
                drop_on_latency=bool(getattr(config, "RTSP_DROP_ON_LATENCY", False)),
                read_timeout_seconds=float(getattr(config, "RTSP_READ_TIMEOUT_SECONDS", 1.5)),
                startup_timeout_seconds=float(getattr(config, "RTSP_STARTUP_TIMEOUT_SECONDS", 10.0)),
            )
        return _OpenCVCapture(self.video_source)

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
                if ret and frame is not None:
                    print("[VIDEO] RTSP reconnected.")
                    return True, frame, datetime.now(IST)
            except Exception as e:
                print(f"[VIDEO] Reconnect failed: {e}")
        return False, None, None

    def _update(self):
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
            grab_ist = datetime.now(IST) if ret and frame is not None else None

            if ret and frame is None:
                self.read_timeout_count += 1
                if self.read_timeout_count == 1 or self.read_timeout_count % 30 == 0:
                    print(f"[VIDEO] Read timeout/no sample ({self.read_timeout_count} consecutive).")
                with self._lock:
                    self._latest_frame = None
                    self._latest_frame_ist = None
                    self.ret = True
                self._new_frame_event.set()
                continue

            if self.read_timeout_count:
                print(f"[VIDEO] Frames restored after {self.read_timeout_count} read timeout(s).")
                self.read_timeout_count = 0

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
        if not self.running:
            return False, None, None

        if not self._is_rtsp:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                return False, None, None
            
            self.current_frame_idx += 1
            self.last_frame_time = time.time()
            if self._latest_frame_ist is None:
                self._latest_frame_ist = datetime.now(IST)
            else:
                self._latest_frame_ist += timedelta(seconds=1.0 / max(self.fps, 1.0))
            self.frame_ist = self._latest_frame_ist
            return True, frame, self._latest_frame_ist

        if block:
            got_new = self._new_frame_event.wait(timeout)
            if not got_new:
                return self.ret, None, None

        with self._lock:
            frame = self._latest_frame
            grab_ist = self._latest_frame_ist
            self._new_frame_event.clear()

        if frame is not None:
            self.current_frame_idx += 1
            self.last_frame_time = time.time()
            self.frame_ist = grab_ist
            return True, frame, grab_ist
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
            "backend": "pyav" if self._is_rtsp else "opencv",
            "reconnect_count": self.reconnect_count,
            "read_timeouts": self.read_timeout_count,
            "frame_idx": self.current_frame_idx,
        }

    def release(self):
        self.running = False
        self._new_frame_event.set()
        try:
            self.cap.release()
        except Exception:
            pass
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
