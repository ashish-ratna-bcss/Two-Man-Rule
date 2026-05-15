import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch
import config
from logic.state_machine import DualAuthStateMachine


def _make_sm(fps=10):
    roi = MagicMock()
    roi.point_in_roi.return_value = False
    roi.get_roi.return_value = None
    return DualAuthStateMachine(roi_manager=roi, fps=fps)


def _pose(head_in_door=True, has_lock_contact=True):
    return {
        "qualified": False,
        "head_in_interaction": False,
        "feet_in_standing": False,
        "waist_near_door": False,
        "ear_order_correct": True,
        "shoulder_order_correct": True,
        "in_locks_roi": False,
        "arms_raised": False,
        "left_right_order": True,
        "has_lock_contact": has_lock_contact,
        "head_in_door": head_in_door,
        "shoulders_in_door": False,
        "anchor": (100, 100),
    }


def _tick(sm, track_id, pose_override, frame_step=1):
    tracked = {track_id: {"keypoints": None, "bbox": None}}
    with patch.object(sm, "_evaluate_unlock_pose", return_value=pose_override):
        sm.update_timers(tracked, frame_step=frame_step)


def _seed(sm, track_id=1):
    sm.session["id_a"] = track_id
    sm.session["id_b"] = None
    sm.session["id_a_left_zone"] = True
    sm.verified_anchors["a"] = (100, 100)


def test_no_violation_before_threshold():
    sm = _make_sm(fps=10)  # min_unlock_frames = 50
    _seed(sm)
    p = _pose(head_in_door=True, has_lock_contact=True)
    for _ in range(49):
        _tick(sm, 1, p)
    assert sm.session["violation_type"] is None
    assert sm.session["same_id_return_timer_frames"] == 49


def test_violation_fires_at_threshold():
    sm = _make_sm(fps=10)
    _seed(sm)
    p = _pose(head_in_door=True, has_lock_contact=True)
    for _ in range(50):
        _tick(sm, 1, p)
    assert sm.session["violation_type"] == "SAME_ID"


def test_timer_pauses_during_grace_not_reset():
    sm = _make_sm(fps=10)
    _seed(sm)
    contact = _pose(head_in_door=True, has_lock_contact=True)
    no_contact = _pose(head_in_door=True, has_lock_contact=False)
    for _ in range(20):
        _tick(sm, 1, contact)
    assert sm.session["same_id_return_timer_frames"] == 20
    grace = config.GRACE_BUFFER_FRAMES
    for _ in range(grace - 1):
        _tick(sm, 1, no_contact)
    assert sm.session["same_id_return_timer_frames"] == 20
    assert sm.session["violation_type"] is None


def test_timer_resets_after_grace_expires():
    sm = _make_sm(fps=10)
    _seed(sm)
    contact = _pose(head_in_door=True, has_lock_contact=True)
    no_contact = _pose(head_in_door=True, has_lock_contact=False)
    for _ in range(20):
        _tick(sm, 1, contact)
    grace = config.GRACE_BUFFER_FRAMES
    for _ in range(grace + 1):
        _tick(sm, 1, no_contact)
    assert sm.session["same_id_return_timer_frames"] == 0
    assert sm.session["violation_type"] is None


def test_no_violation_when_no_lock_contact():
    sm = _make_sm(fps=10)
    _seed(sm)
    p = _pose(head_in_door=True, has_lock_contact=False)
    for _ in range(100):
        _tick(sm, 1, p)
    assert sm.session["violation_type"] is None
    assert sm.session["same_id_return_timer_frames"] == 0


def test_no_violation_when_id_b_already_set():
    sm = _make_sm(fps=10)
    sm.session["id_a"] = 1
    sm.session["id_b"] = 2
    sm.session["id_a_left_zone"] = True
    p = _pose(head_in_door=True, has_lock_contact=True)
    for _ in range(100):
        _tick(sm, 1, p)
    assert sm.session["violation_type"] is None
