# io/visualizer.py
import cv2
import numpy as np
from typing import Dict, Tuple
import math
import config

class Visualizer:
    """Render overlays: bboxes, progress bars, status text."""

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.text_thickness = 2

    def draw_bounding_box(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        color: Tuple[int, int, int],
        label: str = ""
    ) -> np.ndarray:
        """Draw bounding box on frame."""
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if label:
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                self.font,
                self.font_scale,
                color,
                self.text_thickness
            )

        return frame

    def draw_circular_progress_bar(
        self,
        frame: np.ndarray,
        center: Tuple[int, int],
        progress_percent: float,
        radius: int = None,
        color: Tuple[int, int, int] = config.COLOR_AUTHORIZED
    ) -> np.ndarray:
        """Draw circular progress bar (0-100%)."""
        radius = radius or config.PROGRESS_BAR_RADIUS
        progress_percent = min(100, max(0, progress_percent))

        # Draw background circle
        cv2.circle(frame, center, radius, (50, 50, 50), -1)

        # Draw progress arc
        start_angle = -90
        end_angle = start_angle + (progress_percent / 100.0) * 360

        cv2.ellipse(
            frame,
            center,
            (radius, radius),
            0,
            start_angle,
            end_angle,
            color,
            config.PROGRESS_BAR_THICKNESS
        )

        # Draw percentage text
        text = f"{int(progress_percent)}%"
        text_size = cv2.getTextSize(text, self.font, self.font_scale, 1)[0]
        text_pos = (
            center[0] - text_size[0] // 2,
            center[1] + text_size[1] // 2
        )
        cv2.putText(
            frame,
            text,
            text_pos,
            self.font,
            self.font_scale,
            color,
            self.text_thickness
        )

        return frame

    def draw_keypoint(
        self,
        frame: np.ndarray,
        x: float,
        y: float,
        confidence: float,
        color: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 5
    ) -> np.ndarray:
        """Draw a single keypoint with confidence-based sizing."""
        if confidence < 0.3:
            return frame

        radius = max(2, int(radius * confidence))
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)
        return frame

    def draw_status_text(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """Draw text with background box."""
        text_size = cv2.getTextSize(text, self.font, self.font_scale, 1)[0]
        x, y = position

        # Background box
        cv2.rectangle(
            frame,
            (x - 5, y - text_size[1] - 5),
            (x + text_size[0] + 5, y + 5),
            bg_color,
            -1
        )

        # Text
        cv2.putText(
            frame,
            text,
            position,
            self.font,
            self.font_scale,
            color,
            self.text_thickness
        )

        return frame

    def draw_client_overlays(
        self,
        frame: np.ndarray,
        unlocker_labels: dict,
        tracked_persons: dict,
        auth_result: dict,
        is_door_open: bool,
        color_unlocking: Tuple[int, int, int] = None,
        color_authorized: Tuple[int, int, int] = None,
        persons_auth_status=None,
    ) -> np.ndarray:
        """Draw client-facing overlays: unlocker bboxes + 2 status panels only.

        persons_auth_status: None = blank (no transaction yet), True = Available, False = Unavailable
        """
        import config as _cfg
        col_auth = color_authorized or _cfg.COLOR_AUTHORIZED
        col_unlock = color_unlocking or _cfg.COLOR_UNLOCKING

        # Unlocker bounding boxes
        for track_id, label in unlocker_labels.items():
            person = tracked_persons.get(track_id)
            if person is None:
                continue
            bbox = person.get("bbox")
            if bbox is None:
                continue
            color = col_auth if auth_result.get("authorized") else col_unlock
            self.draw_bounding_box(frame, bbox, color, label)

        h, w = frame.shape[:2]
        panel_h = 50
        font = cv2.FONT_HERSHEY_SIMPLEX

        # --- 2 Persons panel (top-left) ---
        # Blank until door transaction completes and auth is evaluated
        panel_w = 380
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
        if persons_auth_status is None:
            cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (80, 80, 80), 2)
            cv2.putText(frame, "2 Persons: --", (20, 10 + panel_h - 13), font, 0.8, (150, 150, 150), 2, cv2.LINE_AA)
        else:
            persons_color = (0, 200, 0) if persons_auth_status else (0, 0, 220)
            persons_text = "2 Persons: Available" if persons_auth_status else "2 Persons: Unavailable"
            cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), persons_color, 2)
            cv2.putText(frame, persons_text, (20, 10 + panel_h - 13), font, 0.8, persons_color, 2, cv2.LINE_AA)

        # --- Door Status panel (top-right) ---
        door_text = "Door: Open" if is_door_open else "Door: Closed"
        door_color = (0, 0, 220) if is_door_open else (0, 200, 0)
        dp_w = 260
        cv2.rectangle(frame, (w - dp_w - 10, 10), (w - 10, 10 + panel_h), (20, 20, 20), -1)
        cv2.rectangle(frame, (w - dp_w - 10, 10), (w - 10, 10 + panel_h), door_color, 2)
        cv2.putText(frame, door_text, (w - dp_w, 10 + panel_h - 13), font, 0.8, door_color, 2, cv2.LINE_AA)

        return frame

    def draw_roi_polygon(
        self,
        frame: np.ndarray,
        points: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw polygonal ROI outline."""
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=thickness)
        return frame

    def draw_roi_label(
        self,
        frame: np.ndarray,
        name: str,
        points: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """Draw ROI name near its polygon center."""
        center = points.reshape(-1, 2).mean(axis=0).astype(int)
        cv2.putText(
            frame,
            name,
            (int(center[0]), int(center[1])),
            self.font,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )
        return frame
