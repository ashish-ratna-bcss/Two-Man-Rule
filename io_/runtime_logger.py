import json
import os
from datetime import datetime
from datetime import timezone, timedelta
from typing import Any, Dict, Optional


IST = timezone(timedelta(hours=5, minutes=30))


class RuntimeEventLogger:
    """Append structured runtime events to per-run JSONL files with date rollover."""

    def __init__(self, base_dir: str, site_name: str, camera_id: str):
        self.base_dir = base_dir
        self.site_name = str(site_name)
        self.camera_id = str(camera_id)
        self.process_id = os.getpid()

        now_ist = datetime.now(IST)
        self.run_id = now_ist.strftime("%Y%m%d_%H%M%S")
        self._active_date = ""
        self._current_file_path = ""
        self._fh = None
        self._open_for_datetime(now_ist)

    def _date_str(self, ts_ist: datetime) -> str:
        return ts_ist.strftime("%d-%m-%Y")

    def _time_str(self, ts_ist: datetime) -> str:
        return ts_ist.strftime("%H-%M-%S")

    def _open_for_datetime(self, ts_ist: datetime) -> None:
        date_str = self._date_str(ts_ist)
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

        filename = f"run_{self._time_str(ts_ist)}.jsonl"
        self._current_file_path = os.path.join(target_dir, filename)
        self._fh = open(self._current_file_path, "a", encoding="utf-8")
        self._active_date = date_str

    @property
    def current_file_path(self) -> str:
        return self._current_file_path

    def write_event(
        self,
        event_type: str,
        message: str,
        *,
        level: str = "INFO",
        details: Optional[Dict[str, Any]] = None,
        frame_idx: Optional[int] = None,
        ts_ist: Optional[datetime] = None,
    ) -> bool:
        ts_ist = ts_ist if ts_ist is not None else datetime.now(IST)
        try:
            self._open_for_datetime(ts_ist)
            payload = {
                "ts_ist": ts_ist.isoformat(),
                "level": str(level).upper(),
                "site_name": self.site_name,
                "camera_id": self.camera_id,
                "event_type": event_type,
                "message": message,
                "details": details or {},
                "run_id": self.run_id,
                "frame_idx": frame_idx,
                "process_id": self.process_id,
            }
            self._fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
            self._fh.flush()
            return True
        except Exception:
            return False

    def close(self) -> None:
        if self._fh is None:
            return
        try:
            self._fh.flush()
            self._fh.close()
        finally:
            self._fh = None