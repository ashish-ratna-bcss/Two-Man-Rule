"""Tests for the deep appearance Re-ID (Option A) — extractor + tracker plumbing.

Live cross-angle re-link is validated against real footage; here we lock the
deterministic units: graceful fallback, preprocessing, cosine math, and the
embedding gallery / kill-switch behavior in the tracker.
"""
import numpy as np
import pytest

import config
from models.reid_extractor import ReIDExtractor
from models.tracker import PersonTracker


# ----------------------------------------------------------------------------
# ReIDExtractor — graceful fallback + preprocessing
# ----------------------------------------------------------------------------
def test_extractor_unavailable_when_model_missing(tmp_path):
    ext = ReIDExtractor(model_path=str(tmp_path / "does_not_exist.onnx"))
    assert ext.available is False
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    assert ext.embed(frame, [[10, 10, 60, 200]]) is None


def test_extractor_preprocess_shape(tmp_path):
    ext = ReIDExtractor(model_path=str(tmp_path / "missing.onnx"))
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    chw = ext._preprocess(frame, [10, 10, 60, 200])
    assert chw is not None
    assert chw.shape == (3, 256, 128)


def test_extractor_preprocess_rejects_empty_bbox(tmp_path):
    ext = ReIDExtractor(model_path=str(tmp_path / "missing.onnx"))
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    assert ext._preprocess(frame, [50, 50, 50, 50]) is None      # zero area
    assert ext._preprocess(frame, [700, 700, 800, 800]) is None  # outside frame


# ----------------------------------------------------------------------------
# Cosine distance math
# ----------------------------------------------------------------------------
def test_cosine_distance():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    d = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    assert PersonTracker._cos_dist(a, b) == pytest.approx(0.0, abs=1e-6)
    assert PersonTracker._cos_dist(a, c) == pytest.approx(1.0, abs=1e-6)
    assert PersonTracker._cos_dist(a, d) == pytest.approx(2.0, abs=1e-6)


# ----------------------------------------------------------------------------
# Tracker gallery + kill-switch
# ----------------------------------------------------------------------------
def _det(x1, y1, x2, y2, conf=0.9):
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 0.9
    kp[:, 0] = np.linspace(x1, x2, 17)
    kp[:, 1] = np.linspace(y1, y2, 17)
    return {"bbox": [x1, y1, x2, y2], "confidence": conf, "keypoints": kp}


def test_gallery_populated_and_normalized():
    tr = PersonTracker()
    tr._reid_enabled = True
    dets = [_det(10, 10, 60, 210), _det(400, 10, 450, 210)]
    embs = np.array([[3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]], dtype=np.float32)
    # caller normally passes L2-normalized rows; tracker keeps what it's given but
    # re-normalizes on EMA merge. Feed normalized rows here.
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    tracked = tr.update(dets, embeddings=embs)
    assert len(tracked) == 2
    assert len(tr.embedding_gallery) == 2
    for vec in tr.embedding_gallery.values():
        assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-5)


def test_kill_switch_keeps_gallery_empty():
    tr = PersonTracker()
    tr._reid_enabled = False
    dets = [_det(10, 10, 60, 210)]
    embs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    tr.update(dets, embeddings=embs)
    assert tr.embedding_gallery == {}


def test_embeddings_none_is_safe_fallback():
    tr = PersonTracker()
    tr._reid_enabled = True
    dets = [_det(10, 10, 60, 210)]
    tracked = tr.update(dets, embeddings=None)   # no embeddings this frame
    assert len(tracked) == 1
    assert tr.embedding_gallery == {}            # nothing to store, no crash
