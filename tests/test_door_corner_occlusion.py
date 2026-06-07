import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock

import cv2
import numpy as np

from models.door_verifier import DoorVerifier


def _make_reference_image(tmp_path):
    reference = np.full((100, 100, 3), 10, dtype=np.uint8)
    cv2.rectangle(reference, (20, 20), (80, 80), (30, 30, 30), -1)
    image_path = tmp_path / "reference.jpg"
    cv2.imwrite(str(image_path), reference)
    return image_path


def _make_verifier(tmp_path):
    roi = np.array([(20, 20), (80, 20), (80, 80), (20, 80)], dtype=np.int32)
    return DoorVerifier(
        reference_image_path=str(_make_reference_image(tmp_path)),
        door_corner_roi=roi,
        similarity_threshold=0.8,
        debounce_threshold=1,
        intensity_threshold=1,
        motion_threshold=1,
        darkening_protection=False,
    )


def test_clear_roi_reaches_verification(tmp_path, monkeypatch):
    verifier = _make_verifier(tmp_path)
    frame = np.full((100, 100, 3), 100, dtype=np.uint8)
    calls = []

    def _capture(curr_patch, reference_patch, visible_mask, ts_ist=None):
        calls.append((curr_patch.copy(), reference_patch.copy()))
        return True

    monkeypatch.setattr(verifier, "_run_verification", _capture)

    result = verifier.verify(frame, tracked_persons={})

    assert result is True
    assert len(calls) == 1
    assert verifier.last_visible_ratio == 1.0
    assert calls[0][0][30, 30] == 100


def test_partial_occlusion_uses_remaining_area(tmp_path, monkeypatch):
    verifier = _make_verifier(tmp_path)
    frame = np.full((100, 100, 3), 100, dtype=np.uint8)
    calls = []

    def _mock_ssim(im1, im2, **kwargs):
        calls.append((im2.copy(), im1.copy()))
        return 0.95

    monkeypatch.setattr("models.door_verifier.ssim", _mock_ssim)

    tracked_persons = {
        1: {"bbox": (20, 20, 42, 80)},
    }

    result = verifier.verify(frame, tracked_persons=tracked_persons)

    assert result is False
    assert len(calls) == 1
    assert verifier.last_visible_ratio >= 0.5
    assert calls[0][0][30, 30] == calls[0][1][30, 30]
    assert np.count_nonzero(calls[0][0] != calls[0][1]) > 0


def test_below_threshold_freezes_stable_state(tmp_path, monkeypatch):
    verifier = _make_verifier(tmp_path)
    verifier.stable_is_open = True
    frame = np.full((100, 100, 3), 100, dtype=np.uint8)
    run_calls = MagicMock(side_effect=AssertionError("_run_verification should not be called"))
    monkeypatch.setattr(verifier, "_run_verification", run_calls)

    tracked_persons = {
        1: {"bbox": (20, 20, 74, 80)},
    }

    result = verifier.verify(frame, tracked_persons=tracked_persons)

    assert result is True
    assert run_calls.call_count == 0
    assert verifier.last_visible_ratio < 0.5