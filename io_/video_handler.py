import cv2
import time
import numpy as np
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
import config

IST = timezone(timedelta(hours=5, minutes=30))


def _quote_gst_value(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_gstreamer_rtsp_pipeline(
    rtsp_url: str,
    *,
    transport: str = "tcp",
    latency_ms: int = 1000,
    drop_on_latency: bool = False,
) -> str:
    """Build the audit-mode RTSP pipeline used by the GStreamer backend."""
    transport = str(transport or "tcp").lower()
    if transport not in {"tcp", "udp", "udp-mcast", "http", "tls"}:
        raise ValueError(f"Unsupported RTSP transport: {transport}")

    drop = "true" if drop_on_latency else "false"
    latency_ms = max(0, int(latency_ms))
    return (
        "rtspsrc "
        f"location={_quote_gst_value(rtsp_url)} "
        f"protocols={transport} "
        f"latency={latency_ms} "
        f"drop-on-latency={drop} "
        "! application/x-rtp,media=video "
        "! decodebin "
        "! videoconvert "
        "! video/x-raw,format=BGR "
        "! appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true"
    )


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


class _GStreamerRTSPCapture:
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
        self.rtsp_url = rtsp_url
        self.read_timeout_seconds = max(0.1, float(read_timeout_seconds))
        self.pipeline_description = build_gstreamer_rtsp_pipeline(
            rtsp_url,
            transport=transport,
            latency_ms=latency_ms,
            drop_on_latency=drop_on_latency,
        )
        self._pending_frame = None
        self._fps = 0.0
        self._width = 0
        self._height = 0
        self._total_frames = 0
        self._gst_error = None

        self.Gst = self._load_gstreamer()
        self.pipeline = self.Gst.parse_launch(self.pipeline_description)
        self.appsink = self.pipeline.get_by_name("appsink")
        if self.appsink is None:
            raise RuntimeError("GStreamer appsink not found in RTSP pipeline.")

        self.bus = self.pipeline.get_bus()
        state_ret = self.pipeline.set_state(self.Gst.State.PLAYING)
        if state_ret == self.Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to set GStreamer RTSP pipeline to PLAYING.")

        self._prime_first_frame(max(0.5, float(startup_timeout_seconds)))

    def _load_gstreamer(self):
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception as e:
            raise RuntimeError(
                "RTSP ingest is configured for GStreamer, but PyGObject/GStreamer "
                "is unavailable. Install system GStreamer plugins plus PyGObject."
            ) from e

        if not Gst.is_initialized():
            Gst.init(None)
        return Gst

    def _prime_first_frame(self, startup_timeout_seconds: float):
        deadline = time.time() + startup_timeout_seconds
        last_error = None
        while time.time() < deadline:
            ret, frame = self.read(timeout_seconds=min(self.read_timeout_seconds, max(0.1, deadline - time.time())))
            if ret and frame is not None:
                self._pending_frame = frame
                return
            if not ret:
                last_error = self._gst_error
                break
        raise RuntimeError(
            "GStreamer RTSP pipeline started but did not deliver a frame "
            f"within {startup_timeout_seconds:.1f}s."
            + (f" Last error: {last_error}" if last_error else "")
        )

    def _pop_error(self):
        msg = self.bus.pop_filtered(self.Gst.MessageType.ERROR | self.Gst.MessageType.EOS)
        if msg is None:
            return None
        if msg.type == self.Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            return f"{err}; debug={debug}"
        if msg.type == self.Gst.MessageType.EOS:
            return "end-of-stream"
        return None

    def read(self, timeout_seconds: Optional[float] = None):
        if self._pending_frame is not None:
            frame = self._pending_frame
            self._pending_frame = None
            return True, frame

        error = self._pop_error()
        if error is not None:
            self._gst_error = error
            return False, None

        timeout = self.read_timeout_seconds if timeout_seconds is None else max(0.1, float(timeout_seconds))
        sample = self.appsink.emit("try-pull-sample", int(timeout * self.Gst.SECOND))
        if sample is None:
            error = self._pop_error()
            if error is not None:
                self._gst_error = error
                return False, None
            return True, None

        frame = self._sample_to_bgr(sample)
        return True, frame

    def _sample_to_bgr(self, sample):
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        fps_value = structure.get_value("framerate")
        if fps_value is not None and getattr(fps_value, "denom", 0):
            self._fps = float(fps_value.num) / float(fps_value.denom)
        self._width = width
        self._height = height

        buf = sample.get_buffer()
        success, map_info = buf.map(self.Gst.MapFlags.READ)
        if not success:
            raise RuntimeError("Failed to map GStreamer sample buffer.")
        try:
            frame = np.ndarray((height, width, 3), dtype=np.uint8, buffer=map_info.data)
            return frame.copy()
        finally:
            buf.unmap(map_info)

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return self._fps or config.DEFAULT_FPS
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self._width
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._height
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return self._total_frames
        return 0

    def release(self):
        try:
            self.pipeline.set_state(self.Gst.State.NULL)
        except Exception:
            pass


class VideoHandler:
    """
    Asynchronous video input with audit-grade RTSP ingest.

    RTSP sources are decoded through GStreamer with a TCP jitter buffer. Local
    video files remain on OpenCV so offline replay behavior is unchanged.
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

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _open(self):
        if self._is_rtsp:
            backend = str(getattr(config, "RTSP_INGEST_BACKEND", "gstreamer")).lower()
            if backend != "gstreamer":
                raise RuntimeError(
                    "RTSP ingest is audit-mode GStreamer only. "
                    f"Unsupported RTSP_INGEST_BACKEND={backend!r}."
                )
            return _GStreamerRTSPCapture(
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
            "backend": "gstreamer" if self._is_rtsp else "opencv",
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
