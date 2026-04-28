# models/pose_detector.py
from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Tuple, Optional
import config

class PoseDetector:
    """YOLOv8-Pose wrapper for skeleton detection."""

    def __init__(self, model_path: str = None, device: str = "auto", half: bool = True):
        """Load YOLOv8-Pose model on GPU if available."""
        import torch
        model_path = model_path or config.YOLO_POSE_MODEL
        if device == "auto":
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            if self.device == "cuda" and not torch.cuda.is_available():
                print("[PoseDetector] CUDA requested but unavailable; falling back to CPU")
                self.device = "cpu"

        self.use_half = bool(half and self.device == "cuda")
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        print(f"[PoseDetector] Device: {self.device}")
        if self.device == 'cuda':
            print(f"[PoseDetector] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[PoseDetector] Half precision: {self.use_half}")
        self.model = YOLO(model_path)
        self.model.to(self.device)  # Move model weights to GPU
        print(f"[PoseDetector] Model loaded on {self.device}")
        self.last_results = None

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run pose detection on frame using GPU.

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
        results = self.model(frame, conf=0.5, verbose=False, device=self.device, half=self.use_half)
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
