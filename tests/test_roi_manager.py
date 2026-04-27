# tests/test_roi_manager.py
import pytest
import numpy as np
from logic.roi_manager import ROIManager

def test_rect_roi_inside():
    rm = ROIManager()
    rm.register_rect_roi("test", 10, 10, 100, 100)
    assert rm.point_in_rect_roi("test", 50, 50) == True

def test_rect_roi_outside():
    rm = ROIManager()
    rm.register_rect_roi("test", 10, 10, 100, 100)
    assert rm.point_in_rect_roi("test", 5, 50) == False

def test_polygon_roi_inside():
    rm = ROIManager()
    points = [(0, 0), (100, 0), (100, 100), (0, 100)]
    rm.register_polygon_roi("test", points)
    assert rm.point_in_polygon_roi("test", 50, 50) == True

def test_polygon_roi_outside():
    rm = ROIManager()
    points = [(0, 0), (100, 0), (100, 100), (0, 100)]
    rm.register_polygon_roi("test", points)
    assert rm.point_in_polygon_roi("test", 150, 50) == False

def test_distance_to_roi():
    rm = ROIManager()
    rm.register_rect_roi("test", 10, 10, 100, 100)
    dist = rm.distance_to_rect_roi("test", 5, 15)
    assert dist == pytest.approx(5.0, abs=0.1)

def test_roi_center():
    rm = ROIManager()
    rm.register_rect_roi("test", 10, 10, 100, 100)
    center = rm.get_roi_center("test")
    assert center == (60, 60)
