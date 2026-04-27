# tests/test_kinematic_fallback.py
import pytest
import numpy as np
from logic.kinematic_fallback import KinematicFallback
import config

def test_wrist_visible():
    kf = KinematicFallback()
    keypoints = np.array([
        [0, 0, 0.1],  # Low confidence
        [10, 10, 0.9]  # High confidence
    ])

    assert kf.is_wrist_visible(keypoints, 0) == False
    assert kf.is_wrist_visible(keypoints, 1) == True

def test_shoulder_static():
    kf = KinematicFallback()

    result1 = kf.is_shoulder_static(1, (50, 50))
    result2 = kf.is_shoulder_static(1, (50, 50))
    result3 = kf.is_shoulder_static(1, (50, 50))

    assert result3 == True  # Should be static

def test_shoulder_moving():
    kf = KinematicFallback()

    kf.is_shoulder_static(1, (50, 50))
    kf.is_shoulder_static(1, (60, 60))
    result = kf.is_shoulder_static(1, (100, 100))

    assert result == False  # Should be moving

def test_euclidean_distance():
    kf = KinematicFallback()
    dist = kf.distance_euclidean((0, 0), (3, 4))
    assert dist == pytest.approx(5.0)
