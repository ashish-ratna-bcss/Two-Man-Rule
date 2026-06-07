# models/reid_extractor.py
"""Deep appearance Re-ID feature extractor (OSNet ONNX).

Produces an L2-normalized embedding per person crop so the tracker can re-link the
same physical person across camera angle and long gaps — where the pose-keypoint
matcher fails. Same uniform / multi-angle is exactly what a cross-view Re-ID model
handles better than a color or geometry descriptor.

Designed to degrade gracefully: if the weights file is missing or the onnxruntime
session cannot be built, ``available`` is False and ``embed`` returns None, so the
tracker transparently falls back to its existing pose-keypoint Re-ID.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import cv2
import numpy as np

import config

# OSNet expects 256x128 RGB, ImageNet-normalized.
_INPUT_W = 128
_INPUT_H = 256
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class ReIDExtractor:
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        self.available = False
        self._session = None
        self._input_name = None
        self.model_path = model_path or getattr(config, "REID_MODEL_PATH", "")

        if not getattr(config, "REID_APPEARANCE_ENABLED", False):
            print("[REID] Appearance Re-ID disabled by config; pose-keypoint fallback in use.")
            return
        if not self.model_path or not os.path.exists(self.model_path):
            print(f"[REID] Model not found at {self.model_path!r}; pose-keypoint fallback in use.")
            return
        try:
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]
            if device != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(self.model_path, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            self.available = True
            print(f"[REID] OSNet loaded ({self.model_path}) providers={providers}")
        except Exception as e:  # pragma: no cover - depends on runtime/model
            print(f"[REID] Failed to load model ({e}); pose-keypoint fallback in use.")
            self._session = None
            self.available = False

    def _preprocess(self, frame: np.ndarray, bbox: Sequence[float]) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2])); y2 = min(h, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        chw = (chw - _MEAN) / _STD
        return chw

    def embed(self, frame: np.ndarray, bboxes: List[Sequence[float]]) -> Optional[np.ndarray]:
        """Return an (N, D) array of L2-normalized embeddings aligned to ``bboxes``.

        Rows for crops that could not be built are zero vectors (treated as "no
        embedding" by the tracker). Returns None when the extractor is unavailable.
        """
        if not self.available or self._session is None or frame is None or not bboxes:
            return None

        batch = []
        valid_idx = []
        for i, bbox in enumerate(bboxes):
            chw = self._preprocess(frame, bbox)
            if chw is not None:
                batch.append(chw)
                valid_idx.append(i)

        out = np.zeros((len(bboxes), 0), dtype=np.float32)
        if not batch:
            return out
        try:
            inp = np.stack(batch, axis=0).astype(np.float32)
            feats = self._session.run(None, {self._input_name: inp})[0]
        except Exception as e:  # pragma: no cover
            print(f"[REID] Inference error ({e}); pose-keypoint fallback for this frame.")
            return None

        feats = np.asarray(feats, dtype=np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats = feats / norms

        result = np.zeros((len(bboxes), feats.shape[1]), dtype=np.float32)
        for row, i in enumerate(valid_idx):
            result[i] = feats[row]
        return result

    def release(self) -> None:
        self._session = None
        self.available = False
