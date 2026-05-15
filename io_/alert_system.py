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

    def __init__(self, evidence_dir: str, runtime_logger=None):
        self.evidence_dir = evidence_dir
        self.runtime_logger = runtime_logger

        # Create directories
        os.makedirs(self.evidence_dir, exist_ok=True)

    def log_event(self, event_type: str, details: Dict = None):
        """Log a generic event to runtime logger if configured."""
        if self.runtime_logger is None:
            return
        self.runtime_logger.write_event(
            event_type=event_type,
            message=f"Alert event: {event_type}",
            level="INFO",
            details=details or {},
        )

