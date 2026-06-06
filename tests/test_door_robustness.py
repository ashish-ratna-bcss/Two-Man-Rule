"""Door-verifier robustness: smooth twilight relax + open/close hysteresis.

See docs/superpowers/specs/2026-06-07-door-accuracy-dawn-falseopen-design.md
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from models.door_verifier import DoorVerifier


def _make_verifier(tmp_path, threshold=0.80, hysteresis=0.05):
    ref = np.full((100, 100, 3), 110, dtype=np.uint8)
    p = str(tmp_path / "ref.jpg")
    cv2.imwrite(p, ref)
    corner = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.int32)
    return DoorVerifier(
        p, door_corner_roi=corner,
        similarity_threshold=threshold, debounce_threshold=5,
        open_hysteresis=hysteresis,
    )


# ---- smooth twilight relax: continuous, bounded, NO cliff at 45 ----

def test_relax_factor_bounds(tmp_path):
    v = _make_verifier(tmp_path)
    assert v._lighting_relax_factor(200.0) == 1.0          # bright -> no relax
    assert v._lighting_relax_factor(0.0) == v._RELAX_MIN_FACTOR  # dark -> max relax
    assert v._RELAX_MIN_FACTOR <= v._lighting_relax_factor(44.0) <= 1.0


def test_relax_factor_monotonic(tmp_path):
    v = _make_verifier(tmp_path)
    xs = [10, 20, 30, 44, 50, 60, 80]
    fs = [v._lighting_relax_factor(x) for x in xs]
    assert fs == sorted(fs)                                  # brighter -> higher factor


def test_relax_factor_no_cliff_at_45(tmp_path):
    # The old hard band had a discontinuity at 45 that flipped the door state.
    v = _make_verifier(tmp_path)
    assert abs(v._lighting_relax_factor(44.9) - v._lighting_relax_factor(45.1)) < 0.01


# ---- hysteresis: harder to ENTER open than to return CLOSED ----

def test_hysteresis_marginal_dip_stays_closed(tmp_path):
    v = _make_verifier(tmp_path, threshold=0.80, hysteresis=0.05)
    v.stable_is_open = False
    # SSIM just below the plain threshold but within the hysteresis margin.
    assert v._ssim_indicates_open(0.78, 0.80) is False      # 0.78 > 0.80-0.05 -> stay CLOSED


def test_hysteresis_deep_drop_opens(tmp_path):
    v = _make_verifier(tmp_path, threshold=0.80, hysteresis=0.05)
    v.stable_is_open = False
    assert v._ssim_indicates_open(0.60, 0.80) is True       # clear structural change -> OPEN


def test_hysteresis_returns_closed_on_recovery(tmp_path):
    v = _make_verifier(tmp_path, threshold=0.80, hysteresis=0.05)
    v.stable_is_open = True
    # Once OPEN, a recovery above the plain threshold returns CLOSED.
    assert v._ssim_indicates_open(0.81, 0.80) is False
    assert v._ssim_indicates_open(0.79, 0.80) is True       # still below thresh -> stay open
