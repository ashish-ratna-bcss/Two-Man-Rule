# logic/roi_manager.py
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional

class ROIManager:
    """Handles rectangular and polygonal ROI operations."""

    def __init__(self):
        self.rois = {}  # name -> {"type": "rect"|"polygon", "data": ...}

    def register_rect_roi(self, name: str, x: int, y: int, w: int, h: int):
        """Register a rectangular ROI."""
        self.rois[name] = {
            "type": "rect",
            "x": x, "y": y, "w": w, "h": h
        }

    def register_polygon_roi(self, name: str, points: List[Tuple[int, int]]):
        """Register a polygonal ROI."""
        self.rois[name] = {
            "type": "polygon",
            "points": np.array(points, dtype=np.int32)
        }

    def point_in_rect_roi(self, name: str, x: float, y: float) -> bool:
        """Check if point (x, y) is inside rectangular ROI."""
        if name not in self.rois or self.rois[name]["type"] != "rect":
            return False

        roi = self.rois[name]
        return (roi["x"] <= x < roi["x"] + roi["w"] and
                roi["y"] <= y < roi["y"] + roi["h"])

    def point_in_polygon_roi(self, name: str, x: float, y: float) -> bool:
        """Check if point (x, y) is inside polygonal ROI using cv2.pointPolygonTest."""
        if name not in self.rois or self.rois[name]["type"] != "polygon":
            return False

        roi = self.rois[name]
        result = cv2.pointPolygonTest(roi["points"], (x, y), False)
        return result >= 0  # Inside or on boundary

    def point_in_roi(self, name: str, x: float, y: float) -> bool:
        """Generic check - dispatches to correct method."""
        if name not in self.rois:
            return False

        roi_type = self.rois[name]["type"]
        if roi_type == "rect":
            return self.point_in_rect_roi(name, x, y)
        elif roi_type == "polygon":
            return self.point_in_polygon_roi(name, x, y)
        return False

    def distance_to_rect_roi(self, name: str, x: float, y: float) -> float:
        """Calculate shortest distance from point to rectangular ROI edge."""
        if name not in self.rois or self.rois[name]["type"] != "rect":
            return float('inf')

        roi = self.rois[name]
        x_min, x_max = roi["x"], roi["x"] + roi["w"]
        y_min, y_max = roi["y"], roi["y"] + roi["h"]

        # Clamp point to ROI bounds
        closest_x = np.clip(x, x_min, x_max)
        closest_y = np.clip(y, y_min, y_max)

        return np.sqrt((x - closest_x)**2 + (y - closest_y)**2)

    def get_roi_center(self, name: str) -> Optional[Tuple[float, float]]:
        """Get center coordinates of ROI."""
        if name not in self.rois:
            return None

        roi = self.rois[name]
        if roi["type"] == "rect":
            return (roi["x"] + roi["w"] / 2, roi["y"] + roi["h"] / 2)
        elif roi["type"] == "polygon":
            points = roi["points"]
            return (points[:, 0].mean(), points[:, 1].mean())
        return None
