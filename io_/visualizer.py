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
        panel_w = 400
        panel_h = 100
        margin_x = 20
        margin_y = 20
        
        x1 = w - panel_w - margin_x
        y1 = margin_y
        x2 = w - margin_x
        y2 = margin_y + panel_h
        
        # Background and Border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 25, 25), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Door Status
        door_label = "Door:"
        door_val = "OPEN" if is_door_open else "CLOSED"
        door_color = (0, 0, 255) if is_door_open else (0, 255, 0)
        
        # 2 Persons Status
        persons_label = "2 Persons:"
        if persons_auth_status is None:
            persons_val = "--"
            persons_color = (150, 150, 150)
        else:
            persons_val = "AVAILABLE" if persons_auth_status else "UNAVAILABLE"
            persons_color = (0, 255, 0) if persons_auth_status else (0, 0, 255)
            
        # Draw Labels (Left aligned)
        cv2.putText(frame, door_label, (x1 + 25, y1 + 40), font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)
        cv2.putText(frame, persons_label, (x1 + 25, y1 + 80), font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)
        
        # Draw Values (Aligned to the right of the labels)
        cv2.putText(frame, door_val, (x1 + 180, y1 + 40), font, font_scale, door_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, persons_val, (x1 + 180, y1 + 80), font, font_scale, persons_color, thickness, cv2.LINE_AA)

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

    def draw_ghost_anchor(
        self,
        frame: np.ndarray,
        anchor: Tuple[float, float],
        label: str,
        color: Tuple[int, int, int] = (150, 150, 150)
    ) -> np.ndarray:
        """Draw a 'ghost' crosshair for a lost person at their last known anchor position."""
        ax, ay = map(int, anchor)
        size = 15
        # Draw crosshair
        cv2.line(frame, (ax - size, ay), (ax + size, ay), color, 1)
        cv2.line(frame, (ax, ay - size), (ax, ay + size), color, 1)
        # Draw small circle
        cv2.circle(frame, (ax, ay), 5, color, 1)
        # Draw label
        cv2.putText(
            frame,
            label,
            (ax + 10, ay - 10),
            self.font,
            0.4,
            color,
            1,
            cv2.LINE_AA
        )
        return frame
