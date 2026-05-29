# models/pose_detector.py
from ultralytics import YOLO
import numpy as np
import torch
from typing import List, Dict
import config

class PoseDetector:
    """YOLOv8-Pose wrapper optimized for production CUDA environments."""

    def __init__(self, model_path: str = None, device: str = "auto", half: bool = True):
        """Load YOLOv8-Pose model with warmup and memory optimizations."""
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
            # Enable cuDNN autotuner for production performance stability
            torch.backends.cudnn.benchmark = True
            
            # Optional memory fraction limit (Production Safety)
            if hasattr(config, "MAX_PROCESS_VRAM_FRACTION") and config.MAX_PROCESS_VRAM_FRACTION is not None:
                try:
                    torch.cuda.set_per_process_memory_fraction(config.MAX_PROCESS_VRAM_FRACTION, 0)
                    print(f"[PoseDetector] Memory fraction limited to {config.MAX_PROCESS_VRAM_FRACTION}")
                except Exception as e:
                    print(f"[WARNING] Could not set memory fraction: {e}")

        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"[PoseDetector] Model loaded on {self.device} (Half={self.use_half})")
        
        # Warmup Phase (Production Stability)
        # Prevents first-inference latency spikes and initializes allocator pools
        if self.device == 'cuda':
            print("[PoseDetector] Starting warmup...")
            try:
                with torch.inference_mode():
                    dummy_input = torch.zeros((1, 3, 640, 640), device=self.device)
                    if self.use_half: dummy_input = dummy_input.half()
                    for _ in range(3):
                        self.model(dummy_input, verbose=False)
                torch.cuda.synchronize()
                print("[PoseDetector] Warmup complete.")
            except Exception as e:
                print(f"[WARNING] Warmup failed: {e}")

        self.last_results = None

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run pose detection with inference_mode and OOM recovery.
        """
        try:
            with torch.inference_mode():
                results = self.model(frame, conf=0.5, verbose=False, device=self.device, half=self.use_half)
            return self._process_results(results)
        except torch.cuda.OutOfMemoryError:
            print("[ERROR] CUDA OOM Detected! Attempting recovery...")
            torch.cuda.empty_cache()
            # Retry once after clearing cache
            try:
                with torch.inference_mode():
                    results = self.model(frame, conf=0.5, verbose=False, device=self.device, half=self.use_half)
                return self._process_results(results)
            except Exception as e:
                # Return None (timeout sentinel) so main loop freezes FSM via LKG
                # rather than treating it as a genuine empty scene → false UNAUTHORIZED.
                print(f"[CRITICAL] Recovery failed: {e}")
                return None
        except Exception as e:
            # Same sentinel for any other transient failure — protects auth FSM.
            print(f"[ERROR] Inference failed: {e}")
            return None

    def _process_results(self, results) -> List[Dict]:
        """Convert YOLO results to lightweight detection dictionaries."""
        detections = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is not None and result.keypoints is not None:
            # Batch CPU transfers to minimize sync points
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            boxes_conf = result.boxes.conf.cpu().numpy()
            kpts_xy = result.keypoints.xy.cpu().numpy()
            kpts_conf = result.keypoints.conf.cpu().numpy()

            for i in range(len(boxes_xyxy)):
                kpt_array = np.hstack([
                    kpts_xy[i],
                    kpts_conf[i].reshape(-1, 1)
                ])

                detections.append({
                    "bbox": boxes_xyxy[i],
                    "confidence": float(boxes_conf[i]),
                    "keypoints": kpt_array,
                    "keypoint_names": [
                        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist", "left_hip", "right_hip",
                        "left_knee", "right_knee", "left_ankle", "right_ankle"
                    ]
                })

        self.last_results = detections
        return detections
