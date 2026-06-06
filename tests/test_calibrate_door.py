"""Unit tests for the offline door-threshold calibration math.

See docs/superpowers/specs/2026-06-07-door-accuracy-dawn-falseopen-design.md
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.calibrate_door import parse_door_lines, suggest_threshold, _percentile


def test_parse_door_lines():
    text = (
        "[DOOR] SSIM: 0.762 | Diff: 24.8 | Intensity: 44.0 (Δ9.7) | Stable: CLOSED\n"
        "noise line\n"
        "[DOOR] SSIM: 0.401 | Diff: 88.0 | Intensity: 120.0 (Δ5.0) | Stable: OPEN\n"
    )
    s = parse_door_lines(text)
    assert len(s) == 2
    assert s[0].ssim == 0.762 and s[0].is_open is False
    assert s[1].ssim == 0.401 and s[1].is_open is True
    assert s[0].intensity == 44.0


def test_percentile():
    assert _percentile([], 50) is None
    assert _percentile([0.5], 1) == 0.5
    vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert _percentile(vals, 0) == 0.1
    assert _percentile(vals, 100) == 0.5


def test_suggest_threshold_separable_bands():
    # closed clusters high (~0.95), open clusters low (~0.40) -> threshold between.
    closed = [0.95, 0.96, 0.94, 0.93, 0.97]
    open_ = [0.40, 0.42, 0.38, 0.45, 0.41]
    r = suggest_threshold(closed, open_, current=0.80)
    assert r["overlap"] is False
    assert r["open_ceiling"] < r["suggested"] < r["closed_floor"]


def test_suggest_threshold_overlap_flagged():
    # bands overlap -> cannot separate by a threshold; must be flagged.
    closed = [0.70, 0.72, 0.68]
    open_ = [0.69, 0.71, 0.74]
    r = suggest_threshold(closed, open_)
    assert r["overlap"] is True


def test_suggest_threshold_no_open_samples():
    # bhimavaram-like: only closed samples, floor ~0.75 -> threshold a margin below.
    closed = [0.762, 0.751, 0.752, 0.78, 0.75]
    r = suggest_threshold(closed, [], current=0.80, margin=0.05)
    assert r["open_n"] == 0
    assert r["suggested"] < min(closed)        # below the closed floor
    assert 0.30 <= r["suggested"] <= 0.95
