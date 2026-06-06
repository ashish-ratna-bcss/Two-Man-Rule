"""Hard-reset hinge for the evening OPEN->CLOSE gate.

The evening gate (main.py) calls state_machine.reset_session() at the door
OPEN->CLOSE transition so unlocker detection starts 100% fresh — nothing
accumulated while the door was open may survive. This test pins that
reset_session() actually wipes all unlocker state.

See docs/superpowers/specs/2026-06-06-evening-gate-open-close-design.md
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock
from logic.state_machine import DualAuthStateMachine


def _make_sm(fps=10):
    roi = MagicMock()
    roi.point_in_roi.return_value = False
    roi.get_roi.return_value = None
    return DualAuthStateMachine(roi_manager=roi, fps=fps)


def test_reset_session_wipes_unlocker_state():
    sm = _make_sm()

    # Simulate open-period progress that MUST NOT survive the OPEN->CLOSE reset.
    sm.assign_unlocker_tag(11, "a")
    sm.assign_unlocker_tag(22, "b")
    sm.session["id_a"] = 11
    sm.session["id_b"] = 22
    sm.session["candidate_a"] = 33
    sm.session["timer_a_frames"] = 99
    sm.session["violation_type"] = "SAME_ID"
    sm.verified_anchors["a"] = (100, 100)
    sm.slot_departed["a"] = True

    sm.reset_session()

    assert sm.unlocker_tags == {}
    assert sm.all_unlocker_ids == {"P1_unlocker": set(), "P2_unlocker": set()}
    assert sm.session["id_a"] is None
    assert sm.session["id_b"] is None
    assert sm.session["candidate_a"] is None
    assert sm.session["timer_a_frames"] == 0
    assert sm.session.get("violation_type") is None
    assert sm.verified_anchors["a"] is None
    assert sm.slot_departed["a"] is False
    # After reset, authorization must be back to INCOMPLETE (no slot authorized).
    auth = sm.check_authorization()
    assert auth["authorized"] is False
    assert auth["lock_a_authorized"] is False
    assert auth["lock_b_authorized"] is False
