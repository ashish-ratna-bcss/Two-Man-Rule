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


def test_localized_quadrant_white_block_is_corrupt():
    # One quadrant fully saturated (decode blowout) — global white_ratio ~0.25 stays
    # under the old 0.65 global gate, but a full cell is saturated -> cell detector fires.
    gate = FrameQualityGate(enabled=True, recovery_good_frames=2)
    gate.evaluate(_normal_frame())
    frame = _normal_frame()
    frame[:60, :80] = 255  # top-left quadrant pure white (~25% of frame)

    result = gate.evaluate(frame)

    assert result.status == FrameQualityStatus.CORRUPT
    assert result.usable is False
    assert "white_block" in result.reason


def test_bright_textured_scene_not_flagged_white_block():
    # A bright-but-textured frame (no flat saturated block) must stay GOOD — guards the
    # cell detector against false positives on legitimately bright scenes.
    gate = FrameQualityGate(enabled=True, recovery_good_frames=2)
    gate.evaluate(_normal_frame())
    rng = np.random.default_rng(0)
    frame = np.clip(rng.integers(170, 235, size=(120, 160, 3)), 0, 255).astype(np.uint8)

    result = gate.evaluate(frame)

    assert result.status == FrameQualityStatus.GOOD


def test_adaptive_recovery_lowers_bar_after_storm():
    # A long freeze storm must not demand the full recovery count forever — after a storm
    # the required consecutive good frames drops to the floor so the stream resumes.
    gate = FrameQualityGate(
        enabled=True, recovery_good_frames=20, stale_after_frames=10_000
    )
    import config
    floor = config.FRAME_QUALITY_RECOVERY_GOOD_FRAMES_MIN
    storm = config.FRAME_QUALITY_RECOVERY_STORM_BAD_FRAMES
    corrupt = _normal_frame()
    corrupt[:60, :80] = 255

    for _ in range(storm + 5):
        gate.evaluate(corrupt.copy())

    # Feed exactly `floor` good frames; the last must become usable (bar lowered to floor).
    res = None
    for i in range(floor):
        res = gate.evaluate(_normal_frame(90 + i))
    assert res.status == FrameQualityStatus.GOOD
    assert res.usable is True


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
