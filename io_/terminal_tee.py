import os
import sys
import threading
from datetime import datetime
from datetime import timezone, timedelta


IST = timezone(timedelta(hours=5, minutes=30))


class _DailyRunLogSink:
    """Write plain-text terminal output into per-run files with midnight rollover."""

    def __init__(self, base_dir: str, site_name: str, camera_id: str, file_prefix: str = "run"):
        self.base_dir = base_dir
        self.site_name = str(site_name)
        self.camera_id = str(camera_id)
        self.file_prefix = file_prefix
        self._active_date = ""
        self._fh = None
        self._path = ""
        self._lock = threading.Lock()
        self._open_for_now()

    def _date_str(self, ts_ist: datetime) -> str:
        return ts_ist.strftime("%d-%m-%Y")

    def _time_str(self, ts_ist: datetime) -> str:
        return ts_ist.strftime("%H-%M-%S")

    def _open_for_now(self) -> None:
        now_ist = datetime.now(IST)
        date_str = self._date_str(now_ist)
        if self._fh is not None and self._active_date == date_str:
            return

        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
            finally:
                self._fh = None

        target_dir = os.path.join(self.base_dir, self.site_name, self.camera_id, date_str)
        os.makedirs(target_dir, exist_ok=True)
        filename = f"{self.file_prefix}_{self._time_str(now_ist)}.log"
        self._path = os.path.join(target_dir, filename)
        self._fh = open(self._path, "a", encoding="utf-8")
        self._active_date = date_str

    def write(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._open_for_now()
            self._fh.write(text)
            self._fh.flush()

    def flush(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is None:
                return
            try:
                self._fh.flush()
                self._fh.close()
            finally:
                self._fh = None


class _TeeTextStream:
    """Forward writes to original stream and to log sink."""

    def __init__(self, original_stream, sink: _DailyRunLogSink):
        self._original = original_stream
        self._sink = sink
        self.encoding = getattr(original_stream, "encoding", "utf-8")
        self.errors = getattr(original_stream, "errors", "strict")

    def write(self, text):
        written = self._original.write(text)
        self._sink.write(text)
        return written

    def flush(self):
        self._original.flush()
        self._sink.flush()

    def isatty(self):
        return self._original.isatty()

    def fileno(self):
        return self._original.fileno()


def enable_terminal_capture(base_dir: str, site_name: str, camera_id: str, file_prefix: str = "run"):
    """Mirror sys.stdout/sys.stderr to logs/{site}/{camera}/{date}/{prefix}_*.log.

    Returns a zero-arg restore function.
    """
    sink = _DailyRunLogSink(base_dir=base_dir, site_name=site_name, camera_id=camera_id, file_prefix=file_prefix)
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = _TeeTextStream(original_stdout, sink)
    sys.stderr = _TeeTextStream(original_stderr, sink)

    def _restore():
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            sink.close()

    return _restore
