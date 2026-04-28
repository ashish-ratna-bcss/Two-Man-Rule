# config.py
import numpy as np

# ============ VIDEO & FPS ============
DEFAULT_FPS = 30  # Will be overridden by actual video FPS
GRACE_BUFFER_FRAMES = 15  # 0.5s at 30 FPS

# ============ TIMERS (in seconds) ============
# One valid key-turn/unlock interaction must last at least 6s and no more than 10s.
MIN_UNLOCK_SECONDS = 6.0
MAX_UNLOCK_SECONDS = 10.0
DWELL_THRESHOLD_SECONDS = MIN_UNLOCK_SECONDS

# ============ GEOMETRIC CONSTRAINTS ============
HAND_LOCK_PROXIMITY_PIXELS = 80
ELBOW_LOCK_PROXIMITY_PIXELS = 130  # Elbows can be offset when the unlocker faces away
WRIST_CONFIDENCE_THRESHOLD = 0.5  # Fallback if wrist < 0.5
ARM_KEYPOINT_CONFIDENCE_THRESHOLD = 0.25
HEAD_CONFIDENCE_THRESHOLD = 0.25
DOOR_FACING_ARM_RAISE_PIXELS = 30
LEFT_RIGHT_ORDER_MIN_PIXELS = 25
UNLOCKER_ANCHOR_MATCH_PIXELS = 95

# ============ POSE KEYPOINT INDICES (YOLOv11) ============
# Standard COCO format: 0=Nose, 1=L_Eye, 2=R_Eye, ..., 9=L_Wrist, 10=R_Wrist, 15=L_Ankle, 16=R_Ankle
KEYPOINT_WRIST_LEFT = 9
KEYPOINT_WRIST_RIGHT = 10
KEYPOINT_ELBOW_LEFT = 7
KEYPOINT_ELBOW_RIGHT = 8
KEYPOINT_SHOULDER_LEFT = 5
KEYPOINT_SHOULDER_RIGHT = 6
KEYPOINT_ANKLE_LEFT = 15
KEYPOINT_ANKLE_RIGHT = 16
KEYPOINT_HIP_LEFT = 11
KEYPOINT_HIP_RIGHT = 12

# ============ POSITIONING THRESHOLDS ============
ANKLE_CONFIDENCE_THRESHOLD = 0.3  # Use ankle if conf >= this
HIP_FALLBACK_THRESHOLD = 0.5      # Fall back to hip if ankle < this

# ============ ROI DEFINITIONS ============
LOCK_A_ROI = np.array([(778, 250), (764, 259), (763, 272), (773, 285), (791, 281), (799, 265), (791, 254)], np.int32)

LOCK_B_ROI = np.array([(828, 429), (818, 440), (822, 453), (835, 460), (847, 450), (850, 435), (843, 428)], np.int32)

DOOR_ROI = np.array([(548, 55), (787, 721), (1047, 570), (926, 2), (665, 0)], np.int32)

DOOR_CORNER_ROI = np.array([
    [556, 53], [612, 30], [621, 53], [567, 72]
], np.int32)

# Reference BGR color for the door corner (from closed_ref.jpg)
DOOR_CORNER_REF_COLOR = (55.75, 52.93, 48.84)
DOOR_COLOR_SENSITIVITY = 15.0  # Threshold for opening detection

STANDING_ZONE = np.array([(801, 753), (1067, 597), (1184, 724), (882, 887)], np.int32)

INTERACTION_ZONE = np.array([(272, 160), (1857, 2), (1750, 714), (275, 1392)], np.int32)

# ============ VISUALIZATION ============
COLOR_DETECTED = (255, 0, 0)     # Blue (BGR)
COLOR_UNLOCKING = (0, 255, 255)  # Yellow
COLOR_AUTHORIZED = (0, 255, 0)   # Green
COLOR_VIOLATION = (0, 0, 255)    # Red
PROGRESS_BAR_RADIUS = 30
PROGRESS_BAR_THICKNESS = 3

# ============ SSIM & DOOR VERIFICATION ============
SSIM_THRESHOLD = 0.92
DOOR_VERIFICATION_OCCUPANCY_MAX = 1

# ============ ALERT SYSTEM ============
EVIDENCE_DIR = "logs/evidence"
LOG_DIR = "logs"

# ============ OCCLUSION FALLBACK ============
ANKLE_OCCLUSION_CONFIDENCE_THRESHOLD = 0.3  # Both ankles below this = fallback mode
HEAD_TO_LOCKER_A_MAX_PIXELS = 150  # Max distance from head to LOCKER_A polygon

# ============ BYTETRACK PARAMETERS ============
TRACK_BUFFER = 30
TRACK_THRESH = 0.5

# ============ MODEL PATHS ============
YOLO_POSE_MODEL = "yolov8n-pose.pt"  # Lightweight nano model, ~6.5MB
CLOSED_DOOR_REFERENCE = "closed_ref.jpg"

# ============ CAMERA & ORCHESTRATION CONFIG ============
RTSP_URLS = [{
    "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.108.159:8001/Streaming/Channels/4001",
    "camera_id": "GF-1-CAM-40",
    "site_id": "1",
    "site_name": "somajiguda"
}]

BASE_OUTPUT_DIR = "StrongRoomCheck"

def create_session():
    """Create a fresh session state dict."""
    return {
        "sequence_state": "WAITING_FOR_FIRST_UNLOCKER",
        # Candidates: person currently attempting an unlock (not yet verified)
        "candidate_a": None,
        "candidate_b": None,
        # Official IDs: assigned ONLY after one complete 6-10s lock interaction
        "id_a": None,
        "id_b": None,
        "timer_a_frames": 0,
        "timer_b_frames": 0,
        "timer_a_seconds": 0.0,
        "timer_b_seconds": 0.0,
        "door_opened": False,
        "grace_buffer_a": 0,
        "grace_buffer_b": 0,
        "last_elbow_pos_a": None,
        "last_elbow_pos_b": None,
        "last_shoulder_pos_a": None,
        "last_shoulder_pos_b": None,
        "improper_positioning": None,
        "violation_type": None,
        "auth_success_logged": False,
    }

def calculate_dwell_frames(fps):
    """Convert minimum unlock seconds to frame count based on actual video FPS."""
    return int(DWELL_THRESHOLD_SECONDS * fps)

def calculate_min_unlock_frames(fps):
    """Convert minimum unlock seconds to frame count based on actual video FPS."""
    return int(MIN_UNLOCK_SECONDS * fps)

def calculate_max_unlock_frames(fps):
    """Convert maximum unlock seconds to frame count based on actual video FPS."""
    return int(MAX_UNLOCK_SECONDS * fps)
