import numpy as np
import pytest

import config
from logic.roi_manager import ROIManager
from logic.state_machine import DualAuthStateMachine


@pytest.fixture
def setup():
    roi_manager = ROIManager()
    roi_manager.register_rect_roi("LOCK_A_ROI", 90, 80, 20, 20)
    roi_manager.register_rect_roi("LOCK_B_ROI", 140, 80, 20, 20)
    roi_manager.register_polygon_roi("DOOR_ROI", [(70, 40), (180, 40), (180, 170), (70, 170)])
    roi_manager.register_polygon_roi("STANDING_ZONE", [(50, 170), (220, 170), (220, 240), (50, 240)])
    roi_manager.register_polygon_roi("INTERACTION_ZONE", [(0, 0), (250, 0), (250, 180), (0, 180)])

    fsm = DualAuthStateMachine(roi_manager, fps=10)
    return roi_manager, fsm


def make_person(head=(120, 60), feet=(120, 210), unlock_pose=True):
    keypoints = np.zeros((17, 3), dtype=float)

    for idx in (0, 1, 2, 3, 4):
        keypoints[idx] = [head[0], head[1], 0.9]

    keypoints[config.KEYPOINT_HIP_LEFT] = [115, 165, 0.9]
    keypoints[config.KEYPOINT_HIP_RIGHT] = [125, 165, 0.9]
    keypoints[config.KEYPOINT_ANKLE_LEFT] = [feet[0] - 5, feet[1], 0.9]
    keypoints[config.KEYPOINT_ANKLE_RIGHT] = [feet[0] + 5, feet[1], 0.9]

    if unlock_pose:
        # For this camera, a back-facing person has left-side keypoints on the
        # left of the image and right-side keypoints on the right.
        keypoints[config.KEYPOINT_ELBOW_LEFT] = [100, 90, 0.9]
        keypoints[config.KEYPOINT_WRIST_LEFT] = [100, 90, 0.9]
        keypoints[config.KEYPOINT_ELBOW_RIGHT] = [150, 90, 0.9]
        keypoints[config.KEYPOINT_WRIST_RIGHT] = [150, 90, 0.9]

    return {"bbox": [80, 30, 170, 220], "keypoints": keypoints}


def run_frames(fsm, persons, frames):
    for _ in range(frames):
        fsm.update_occupancy(persons)
        fsm.update_timers(persons)


def test_occupancy_ignores_people_who_are_not_unlocking(setup):
    _, fsm = setup
    persons = {
        1: {"bbox": [50, 50, 75, 75]},
        2: {"bbox": [50, 50, 75, 75]},
        3: {"bbox": [50, 50, 75, 75]},
    }

    result = fsm.update_occupancy(persons)

    assert result == "OK"
    assert len(fsm.active_ids_in_zone) == 0


def test_first_unlocker_is_assigned_after_six_second_pose_and_release(setup):
    _, fsm = setup

    run_frames(fsm, {1: make_person()}, fsm.min_unlock_frames)
    fsm.update_occupancy({})
    fsm.update_timers({})

    assert fsm.session["id_a"] == 1
    assert fsm.session["sequence_state"] == "WAITING_FOR_SECOND_UNLOCKER"


def test_short_unlock_pose_does_not_assign_id(setup):
    _, fsm = setup

    run_frames(fsm, {1: make_person()}, fsm.min_unlock_frames - 1)
    run_frames(fsm, {}, config.GRACE_BUFFER_FRAMES + 1)

    assert fsm.session["id_a"] is None
    assert fsm.session["timer_a_frames"] == 0


def test_dual_auth_requires_two_different_unlockers(setup):
    _, fsm = setup

    run_frames(fsm, {1: make_person()}, fsm.min_unlock_frames)
    run_frames(fsm, {}, 1)
    run_frames(fsm, {2: make_person(head=(130, 60), feet=(130, 210))}, fsm.min_unlock_frames)
    run_frames(fsm, {}, 1)

    result = fsm.check_authorization()

    assert fsm.session["id_a"] == 1
    assert fsm.session["id_b"] == 2
    assert result["authorized"] is True


def test_candidate_survives_raw_tracker_id_change(setup):
    _, fsm = setup

    run_frames(fsm, {10: make_person()}, fsm.min_unlock_frames // 2)
    run_frames(fsm, {99: make_person(head=(122, 60), feet=(122, 210))}, fsm.min_unlock_frames // 2)
    run_frames(fsm, {}, 1)

    assert fsm.session["id_a"] == 99
    assert fsm.session["timer_a_frames"] >= fsm.min_unlock_frames


def test_second_unlocker_can_use_same_standing_spot_with_new_tracker_id(setup):
    _, fsm = setup

    run_frames(fsm, {1: make_person()}, fsm.min_unlock_frames)
    run_frames(fsm, {}, 1)
    run_frames(fsm, {2: make_person(head=(122, 60), feet=(122, 210))}, fsm.min_unlock_frames)
    run_frames(fsm, {}, 1)

    assert fsm.session["id_a"] == 1
    assert fsm.session["id_b"] == 2


def test_reversed_left_right_elbows_do_not_qualify(setup):
    _, fsm = setup
    person = make_person()
    keypoints = person["keypoints"]
    keypoints[config.KEYPOINT_ELBOW_LEFT] = [150, 90, 0.9]
    keypoints[config.KEYPOINT_WRIST_LEFT] = [150, 90, 0.9]
    keypoints[config.KEYPOINT_ELBOW_RIGHT] = [100, 90, 0.9]
    keypoints[config.KEYPOINT_WRIST_RIGHT] = [100, 90, 0.9]

    run_frames(fsm, {1: person}, fsm.min_unlock_frames)

    assert fsm.session["id_a"] is None


def test_same_ids_are_rejected(setup):
    _, fsm = setup
    fsm.session["id_a"] = 1
    fsm.session["id_b"] = 1
    fsm.session["timer_a_frames"] = fsm.min_unlock_frames
    fsm.session["timer_b_frames"] = fsm.min_unlock_frames

    result = fsm.check_authorization()

    assert result["authorized"] is False
    assert result["violation_type"] == "SAME_ID"


def test_occlusion_fallback_maintains_timer_when_ankles_hidden(setup):
    _, fsm = setup

    run_frames(fsm, {1: make_person()}, fsm.min_unlock_frames)
    run_frames(fsm, {}, 1)

    assert fsm.session["id_a"] == 1
    assert fsm.session["sequence_state"] == "WAITING_FOR_SECOND_UNLOCKER"
    timer_at_assignment = fsm.session["timer_a_frames"]

    person_occluded = make_person()
    keypoints = person_occluded["keypoints"]
    keypoints[config.KEYPOINT_ANKLE_LEFT][2] = 0.2
    keypoints[config.KEYPOINT_ANKLE_RIGHT][2] = 0.2

    run_frames(fsm, {1: person_occluded}, 3)

    timer_after_occlusion = fsm.session["timer_a_frames"]
    assert timer_after_occlusion >= timer_at_assignment, "Timer should continue during occlusion fallback"
