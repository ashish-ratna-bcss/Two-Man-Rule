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

    def log_event(self, event_type: str, details: Dict = None):
        """Log a generic event (printing only since session_log is dead)."""
        # In a real system, this would push to a database or cloud logger.
        pass

