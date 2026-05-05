# logic/roi_manager.py
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional

class ROIManager:
    """Handles polygonal ROI operations."""

    def __init__(self):
        self.rois = {}  # name -> {"points": np.ndarray}

    def register_polygon_roi(self, name: str, points: List[Tuple[int, int]]):
        """Register a polygonal ROI."""
        self.rois[name] = {
            "points": np.array(points, dtype=np.int32)
        }

    def point_in_polygon_roi(self, name: str, x: float, y: float) -> bool:
        """Check if point (x, y) is inside polygonal ROI using cv2.pointPolygonTest."""
        if name not in self.rois:
            return False

        roi = self.rois[name]
        result = cv2.pointPolygonTest(roi["points"], (x, y), False)
        return result >= 0  # Inside or on boundary

    def point_in_roi(self, name: str, x: float, y: float) -> bool:
        """Generic check - dispatches to point_in_polygon_roi."""
        return self.point_in_polygon_roi(name, x, y)

    def get_roi_center(self, name: str) -> Optional[Tuple[float, float]]:
        """Get center coordinates of ROI."""
        if name not in self.rois:
            return None

        roi = self.rois[name]
        points = roi["points"]
        return (points[:, 0].mean(), points[:, 1].mean())

    def get_roi(self, name: str) -> Optional[np.ndarray]:
        """Get polygon points array for ROI."""
        if name not in self.rois:
            return None
        roi = self.rois[name]
        if roi["type"] == "polygon":
            return roi["points"]
        return None
