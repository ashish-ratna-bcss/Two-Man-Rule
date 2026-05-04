# io/alert_system.py
import cv2
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, Optional
import config

class AlertSystem:
    """Handle screenshot capture and event logging."""

    def __init__(self, evidence_dir: str = None):
        self.evidence_dir = evidence_dir or config.EVIDENCE_DIR

        # Create directories
        os.makedirs(self.evidence_dir, exist_ok=True)

        self.session_log = []

    def _timestamp_string(self) -> str:
        """Return ISO timestamp string."""
        return datetime.now().isoformat()

    def _filename_timestamp(self) -> str:
        """Return timestamp for filenames (replace colons)."""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def capture_success(self, frame: np.ndarray):
        """Capture SUCCESS state: both locks authorized by different IDs."""
        filename = f"SUCCESS_{self._filename_timestamp()}.jpg"
        filepath = os.path.join(self.evidence_dir, filename)
        cv2.imwrite(filepath, frame)

        event = {
            "timestamp": self._timestamp_string(),
            "event_type": "SUCCESS",
            "filename": filename,
            "filepath": filepath
        }
        self.session_log.append(event)
        return filepath

    def capture_violation_solo(self, frame: np.ndarray):
        """Capture VIOLATION: door opened with only 1 person authorized."""
        filename = f"VIOLATION_SOLO_{self._filename_timestamp()}.jpg"
        filepath = os.path.join(self.evidence_dir, filename)
        cv2.imwrite(filepath, frame)

        event = {
            "timestamp": self._timestamp_string(),
            "event_type": "VIOLATION_SOLO",
            "filename": filename,
            "filepath": filepath
        }
        self.session_log.append(event)
        return filepath

    def capture_violation_overcrowd(self, frame: np.ndarray):
        """Capture VIOLATION: door opened with >2 people in zone."""
        filename = f"VIOLATION_OVERCROWD_{self._filename_timestamp()}.jpg"
        filepath = os.path.join(self.evidence_dir, filename)
        cv2.imwrite(filepath, frame)

        event = {
            "timestamp": self._timestamp_string(),
            "event_type": "VIOLATION_OVERCROWD",
            "filename": filename,
            "filepath": filepath
        }
        self.session_log.append(event)
        return filepath

    def log_event(self, event_type: str, details: Dict = None):
        """Log a generic event."""
        event = {
            "timestamp": self._timestamp_string(),
            "event_type": event_type,
            "details": details or {}
        }
        self.session_log.append(event)

