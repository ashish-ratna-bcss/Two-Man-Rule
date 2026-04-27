# models/pose_detector.py
from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Tuple, Optional
import config

class PoseDetector:
    """YOLOv11-Pose wrapper for skeleton detection."""

    def __init__(self, model_path: str = None):
        """Load YOLOv11-Pose model."""
        model_path = model_path or config.YOLO_POSE_MODEL
        self.model = YOLO(model_path)
        self.last_results = None

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run pose detection on frame.

        Returns:
            List of detections: [
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                    "keypoints": np.array([[x, y, conf], ...], shape=(17, 3)),
                    "keypoint_names": ["nose", "left_eye", ...]
                },
                ...
            ]
        """
        results = self.model(frame, conf=0.5, verbose=False)
        detections = []

        if results and len(results) > 0:
            result = results[0]

            # Iterate over detected persons
            if result.boxes is not None and result.keypoints is not None:
                for i, (box, kpts) in enumerate(zip(result.boxes, result.keypoints)):
                    bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0].cpu().numpy())

                    # Keypoints shape: (17, 3) - [x, y, confidence]
                    keypoints = kpts.xy[0].cpu().numpy()  # First (x, y) pairs
                    confidences = kpts.conf[0].cpu().numpy()  # Confidence for each

                    # Combine into (17, 3) array
                    kpt_array = np.hstack([
                        keypoints,
                        confidences.reshape(-1, 1)
                    ])

                    detections.append({
                        "bbox": bbox,
                        "confidence": conf,
                        "keypoints": kpt_array,  # (17, 3)
                        "keypoint_names": [
                            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                            "left_wrist", "right_wrist", "left_hip", "right_hip",
                            "left_knee", "right_knee", "left_ankle", "right_ankle"
                        ]
                    })

        self.last_results = detections
        return detections

    def get_keypoint(self, detection: Dict, keypoint_idx: int) -> Tuple[float, float, float]:
        """
        Extract single keypoint from detection.

        Returns:
            (x, y, confidence)
        """
        keypoints = detection["keypoints"]
        if keypoint_idx < len(keypoints):
            return tuple(keypoints[keypoint_idx])
        return (0, 0, 0)
