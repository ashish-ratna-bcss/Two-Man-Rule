"""Tests for the 2026-06-07 false-open / evidence fixes.

Covers:
  Fix 1 — bright-washout structural veto in DoorVerifier
  Fix 2 — DoorVerifier.reset_stabilization
  Fix 3/5 — same_id_offender snapshot + appearance gate in the state machine
"""
import os
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from models.door_verifier import DoorVerifier
from logic.state_machine import DualAuthStateMachine

IST = timezone(timedelta(hours=5, minutes=30))


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _textured_patch(w=120, h=80):
    """A patch with real edges/structure so SSIM + gradient-SSIM are meaningful."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = 90
    cv2.rectangle(img, (15, 12), (w - 15, h - 12), (200, 200, 200), 3)
    cv2.line(img, (0, h // 2), (w, h // 2), (160, 160, 160), 2)
    cv2.circle(img, (w // 3, h // 2), 10, (40, 40, 40), -1)
    return img


def _make_verifier(tmp_path, **kw):
    ref = _textured_patch()
    p = os.path.join(tmp_path, "ref.png")
    cv2.imwrite(p, ref)
    roi = np.array([[0, 0], [ref.shape[1] - 1, 0],
                    [ref.shape[1] - 1, ref.shape[0] - 1], [0, ref.shape[0] - 1]],
                   dtype=np.int32)
    v = DoorVerifier(
        reference_image_path=p,
        door_corner_roi=roi,
        similarity_threshold=kw.get("similarity_threshold", 0.6),
        debounce_threshold=kw.get("debounce_threshold", 2),
        motion_threshold=kw.get("motion_threshold", 1.0),
        open_hysteresis=kw.get("open_hysteresis", 0.0),
    )
    return v, ref


def _drive(verifier, frame, n=6, start=None):
    """Feed the same frame n times with advancing timestamps to clear debounce."""
    t0 = start or datetime(2026, 6, 7, 7, 0, 0, tzinfo=IST)
    out = False
    for i in range(n):
        out = verifier.verify(frame, tracked_persons=None, ts_ist=t0 + timedelta(seconds=0.2 * (i + 1)))
    return out


# ----------------------------------------------------------------------------
# Fix 1 — bright-washout structural veto
# ----------------------------------------------------------------------------
def test_washout_does_not_false_open(tmp_path):
    v, ref = _make_verifier(str(tmp_path))
    # stabilize CLOSED on the clean reference first
    _drive(v, ref, n=4)
    assert v.stable_is_open is False

    # Dawn washout: reduced contrast + raised brightness, STRUCTURE preserved.
    washed = np.clip(ref.astype(np.float32) * 0.35 + 160, 0, 255).astype(np.uint8)
    is_open = _drive(v, washed, n=10)
    # large intensity shift → veto active; edges intact → grad-SSIM high → stays CLOSED
    assert v.last_intensity_diff > 25.0
    assert is_open is False
    assert v.stable_is_open is False


def test_real_structural_change_still_opens(tmp_path):
    v, ref = _make_verifier(str(tmp_path))
    _drive(v, ref, n=4)
    assert v.stable_is_open is False

    # Real door opening: structure destroyed (flat dark cavity) → grad-SSIM low → opens
    opened = np.full_like(ref, 30)
    is_open = _drive(v, opened, n=10)
    assert is_open is True
    assert v.stable_is_open is True


# ----------------------------------------------------------------------------
# Fix 2 — reset_stabilization
# ----------------------------------------------------------------------------
def test_reset_stabilization(tmp_path):
    v, ref = _make_verifier(str(tmp_path))
    v.has_stabilized = True
    v.stable_is_open = True
    v.candidate_state = False
    v.candidate_state_start_time = datetime.now(IST)

    v.reset_stabilization()

    assert v.has_stabilized is False
    assert v.stable_is_open is True            # last state preserved
    assert v.candidate_state is True           # candidate re-aligned to stable
    assert v.candidate_state_start_time is None


# ----------------------------------------------------------------------------
# Fix 3 / Fix 5 — appearance gate + offender snapshot
# ----------------------------------------------------------------------------
def _person(color, w=40, h=90, seed=0, jitter=0):
    """A body-like crop: textured (spread histogram) with a dominant clothing color."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    noise = rng.integers(-25, 25, size=(h, w, 3)) + jitter
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_appearance_similarity_same_vs_different():
    red = _person((40, 40, 200), seed=1)
    red_again = _person((40, 40, 200), seed=2, jitter=4)   # same clothing, slight change
    blue = _person((200, 40, 40), seed=3)
    sig_red = DualAuthStateMachine._appearance_signature(red, [0, 0, 40, 90])
    sig_red2 = DualAuthStateMachine._appearance_signature(red_again, [0, 0, 40, 90])
    sig_blue = DualAuthStateMachine._appearance_signature(blue, [0, 0, 40, 90])
    same = DualAuthStateMachine._appearance_similarity(sig_red, sig_red2)
    diff = DualAuthStateMachine._appearance_similarity(sig_red, sig_blue)
    assert same > 0.9
    assert diff < same


def _bare_sm():
    sm = DualAuthStateMachine.__new__(DualAuthStateMachine)
    sm.session = {}
    sm.body_fingerprints = {"a": None, "b": None}
    sm.slot_anchors = {"a": None, "b": None}
    sm.candidate_bbox = {"a": None, "b": None}
    return sm


def test_appearance_gate_blocks_same_person_id_switch(monkeypatch):
    # Gate ships DISABLED for same-uniform deployments; force it on to test the logic.
    import config
    monkeypatch.setattr(config, "APPEARANCE_GATE_ENABLED", True)
    sm = _bare_sm()
    red = _person((40, 40, 200), seed=1)
    sm.body_fingerprints["a"] = {"hsv": DualAuthStateMachine._appearance_signature(red, [0, 0, 40, 90])}
    sm.session["id_a"] = 7
    tracked = {21: {"bbox": [0, 0, 40, 90]}}  # same red person, new id 21

    blocked = sm._appearance_blocks_second_unlocker("b", 21, red, tracked)

    assert blocked is True
    assert sm.session["violation_type"] == "SAME_ID"
    assert sm.session["same_id_offender"] == 7      # Fix 3: offender id captured


def test_appearance_gate_allows_distinct_second_person(monkeypatch):
    import config
    monkeypatch.setattr(config, "APPEARANCE_GATE_ENABLED", True)
    sm = _bare_sm()
    red = _person((40, 40, 200), seed=1)
    blue = _person((200, 40, 40), seed=3)
    sm.body_fingerprints["a"] = {"hsv": DualAuthStateMachine._appearance_signature(red, [0, 0, 40, 90])}
    sm.session["id_a"] = 7
    tracked = {9: {"bbox": [0, 0, 40, 90]}}  # genuinely different (blue) person

    blocked = sm._appearance_blocks_second_unlocker("b", 9, blue, tracked)

    assert blocked is False
    assert sm.session.get("violation_type") != "SAME_ID"
