# tests/test_state_machine.py
import pytest
import numpy as np
from logic.roi_manager import ROIManager
from logic.state_machine import DualAuthStateMachine
import config

@pytest.fixture
def setup():
    roi_manager = ROIManager()
    roi_manager.register_rect_roi("LOCK_A_ROI", 0, 0, 50, 50)
    roi_manager.register_rect_roi("LOCK_B_ROI", 100, 0, 50, 50)
    roi_manager.register_polygon_roi("INTERACTION_ZONE", [(0, 0), (200, 0), (200, 200), (0, 200)])

    fsm = DualAuthStateMachine(roi_manager, fps=30)
    return roi_manager, fsm

def test_occupancy_census_zero(setup):
    roi_manager, fsm = setup
    result = fsm.update_occupancy({})
    assert len(fsm.active_ids_in_zone) == 0
    assert result == "OK"

def test_occupancy_census_overcrowd(setup):
    roi_manager, fsm = setup
    persons = {
        1: {"bbox": [50, 50, 75, 75]},
        2: {"bbox": [50, 50, 75, 75]},
        3: {"bbox": [50, 50, 75, 75]}
    }
    result = fsm.update_occupancy(persons)
    assert result == "VIOLATION_OVERCROWD"
    assert fsm.session["timer_a_frames"] == 0

def test_dual_auth_different_ids(setup):
    roi_manager, fsm = setup
    fsm.session["id_a"] = 1
    fsm.session["id_b"] = 2
    fsm.session["timer_a_frames"] = 300  # 10 seconds at 30 FPS
    fsm.session["timer_b_frames"] = 300

    result = fsm.check_authorization()
    assert result["authorized"] == True
    assert result["lock_a_authorized"] == True
    assert result["lock_b_authorized"] == True

def test_dual_auth_same_ids(setup):
    roi_manager, fsm = setup
    fsm.session["id_a"] = 1
    fsm.session["id_b"] = 1
    fsm.session["timer_a_frames"] = 300
    fsm.session["timer_b_frames"] = 300

    result = fsm.check_authorization()
    assert result["authorized"] == False
    assert result["violation_type"] == "SAME_ID"
