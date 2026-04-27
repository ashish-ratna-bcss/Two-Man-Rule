# tests/test_door_verifier.py
import pytest
import numpy as np
import cv2
from models.door_verifier import DoorVerifier

@pytest.fixture
def create_dummy_image():
    """Create a 100x100 dummy image for testing."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 200

def test_door_verifier_initialization(create_dummy_image, tmp_path):
    img_path = tmp_path / "ref.jpg"
    cv2.imwrite(str(img_path), create_dummy_image)

    dv = DoorVerifier(str(img_path))
    assert dv.reference is not None
    assert dv.threshold == 0.92

def test_door_verifier_closed(create_dummy_image, tmp_path):
    img_path = tmp_path / "ref.jpg"
    cv2.imwrite(str(img_path), create_dummy_image)

    dv = DoorVerifier(str(img_path))
    is_open = dv.is_door_open(create_dummy_image)
    assert is_open == False  # Door closed (high SSIM)
