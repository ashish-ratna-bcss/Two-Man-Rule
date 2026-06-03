import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from io_.frame_quality import FrameQualityGate, FrameQualityStatus
from main import should_freeze_for_frame_quality


def _normal_frame(value=90):
    frame = np.full((120, 160, 3), value, dtype=np.uint8)
    cv2.rectangle(frame, (20, 25), (70, 95), (105, 105, 105), -1)
    cv2.line(frame, (90, 20), (130, 100), (70, 70, 70), 2)
    return frame


def test_normal_stable_frame_is_good():
    gate = FrameQualityGate(enabled=True, recovery_good_frames=2)

    result = gate.evaluate(_normal_frame())

    assert result.status == FrameQualityStatus.GOOD
    assert result.usable is True
    assert should_freeze_for_frame_quality(result, frame_quality_active=True) is False


def test_left_right_white_split_is_corrupt():
    gate = FrameQualityGate(enabled=True, recovery_good_frames=2)
    gate.evaluate(_normal_frame())
    frame = _normal_frame()
    frame[:, :80] = 255

    result = gate.evaluate(frame)

    assert result.status == FrameQualityStatus.CORRUPT
    assert result.usable is False
    assert "brightness_split" in result.reason or "white_block" in result.reason
    assert should_freeze_for_frame_quality(result, frame_quality_active=True) is True


def test_rainbow_chroma_corruption_is_corrupt():
    gate = FrameQualityGate(enabled=True, recovery_good_frames=2)
    gate.evaluate(_normal_frame())
    frame = _normal_frame()
    frame[:, :80, 0] = 255
    frame[:, 40:120, 1] = 255
    frame[:, 80:, 2] = 255

    result = gate.evaluate(frame)

    assert result.status == FrameQualityStatus.CORRUPT
    assert result.usable is False
    assert "chroma" in result.reason


def test_repeated_identical_frames_become_stale():
    gate = FrameQualityGate(enabled=True, stale_after_frames=3, recovery_good_frames=2)
    frame = _normal_frame()

    result = None
    for _ in range(5):
        result = gate.evaluate(frame.copy())

    assert result.status == FrameQualityStatus.STALE
    assert result.usable is False
    assert result.reason == "repeated_identical_frames"


def test_realistic_small_brightness_change_stays_good():
    gate = FrameQualityGate(enabled=True, recovery_good_frames=2)
    result_a = gate.evaluate(_normal_frame(90))
    result_b = gate.evaluate(_normal_frame(100))

    assert result_a.status == FrameQualityStatus.GOOD
    assert result_b.status == FrameQualityStatus.GOOD
    assert result_b.usable is True


def test_recovery_requires_configured_good_frames():
    gate = FrameQualityGate(enabled=True, stale_after_frames=10, recovery_good_frames=2)
    gate.evaluate(_normal_frame())
    corrupt = _normal_frame()
    corrupt[:, :80] = 255

    bad = gate.evaluate(corrupt)
    first_good = gate.evaluate(_normal_frame(91))
    second_good = gate.evaluate(_normal_frame(92))

    assert bad.usable is False
    assert first_good.status == FrameQualityStatus.GOOD
    assert first_good.usable is False
    assert second_good.status == FrameQualityStatus.GOOD
    assert second_good.usable is True
