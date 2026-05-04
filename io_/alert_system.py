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

    def log_event(self, event_type: str, details: Dict = None):
        """Log a generic event to memory."""
        event = {
            "timestamp": self._timestamp_string(),
            "event_type": event_type,
            "details": details or {}
        }
        self.session_log.append(event)

