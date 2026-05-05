# config.py
import numpy as np

# ============ VIDEO & FPS ============
DEFAULT_FPS = 30  # Will be overridden by actual video FPS
GRACE_BUFFER_FRAMES = 15  # 0.5s at 30 FPS
MORNING_POST_OPEN_AUTH_SECONDS = 10.0  # After door opens, wait up to 30s to confirm 2 authenticated persons.
                                        # Increased from 5s to allow the A→B sequential gate to complete
                                        # (both wrists on LOCK_A, then both wrists on LOCK_B) before timeout.

# ============ TIMERS (in seconds) ============
# One valid key-turn/unlock interaction must last at least 6s and no more than 10s.
MIN_UNLOCK_SECONDS = 6.0
MAX_UNLOCK_SECONDS = 10.0
EVENING_SECOND_UNLOCKER_TIMEOUT_SECONDS = 300.0

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
KEYPOINT_EAR_LEFT = 3
KEYPOINT_EAR_RIGHT = 4
KEYPOINT_SHOULDER_LEFT = 5
KEYPOINT_SHOULDER_RIGHT = 6
KEYPOINT_ANKLE_LEFT = 15
KEYPOINT_ANKLE_RIGHT = 16
KEYPOINT_HIP_LEFT = 11
KEYPOINT_HIP_RIGHT = 12

# ============ POSITIONING THRESHOLDS ============
ANKLE_CONFIDENCE_THRESHOLD = 0.3  # Use ankle if conf >= this
HIP_FALLBACK_THRESHOLD = 0.5      # Fall back to hip if ankle < this

# ============ VISUALIZATION ============
COLOR_DETECTED = (255, 0, 0)     # Blue (BGR)
COLOR_UNLOCKING = (0, 255, 255)  # Yellow
COLOR_AUTHORIZED = (0, 255, 0)   # Green
COLOR_VIOLATION = (0, 0, 255)    # Red
PROGRESS_BAR_RADIUS = 30
PROGRESS_BAR_THICKNESS = 3

# ============ SSIM & DOOR VERIFICATION ============
SSIM_THRESHOLD = 0.92

# ============ ALERT SYSTEM ============

# ============ OCCLUSION FALLBACK ============
ANKLE_OCCLUSION_CONFIDENCE_THRESHOLD = 0.3  # Both ankles below this = fallback mode
HEAD_TO_LOCKS_MAX_PIXELS = 150  # Max distance from head to LOCKS_ROI polygon

# ============ BYTETRACK PARAMETERS ============
TRACK_BUFFER = 30
TRACK_THRESH = 0.5

# ============ MODEL PATHS ============
YOLO_POSE_MODEL = "yolov8n-pose.pt"  # Lightweight nano model, ~6.5MB

# ============ STREAMS & ORCHESTRATION CONFIG ============
STREAMS_CONFIG = [
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.108.159:8001/Streaming/Channels/4001",
        "camera_id": "GF-1-CAM-40",
        "site_id": "1",
        "site_name": "somajiguda",
        "closed_door_reference": "closed_GF-1-CAM-40.jpg",
        "ssim_threshold": 0.92,
        "mirror_left_right": False,  # Person faces door (back to top-down cam): right hand IS on video-right, left on video-left
        "rois": {
            "LOCKS_ROI": np.array([(680, 246), (833, 175), (940, 481), (796, 550)], np.int32),
            "DOOR_ROI": np.array([(548, 55), (787, 721), (1047, 570), (926, 2), (665, 0)], np.int32),
            "DOOR_CORNER_ROI": np.array([[640.30, 13.22], [659.51, 73.73], [845.86, 9.38]], np.int32),
            "STANDING_ZONE": np.array([(801, 753), (1067, 597), (1184, 724), (882, 887)], np.int32),
            "INTERACTION_ZONE": np.array([(272, 160), (1857, 2), (1750, 714), (275, 1392)], np.int32)
        }
    },
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.108.159:8002/Streaming/Channels/4201",
        "camera_id": "FF-1-CAM-42",
        "site_id": "1",
        "site_name": "somajiguda",
        "closed_door_reference": "closed_FF-1-CAM-42.jpg",
        "ssim_threshold": 0.88,
        "rois": {
            "LOCKS_ROI": np.array([(333, 247), (809, 150), (876, 558), (433, 683)], np.int32),
            "DOOR_ROI": np.array([(401, 1), (817, 0), (891, 638), (548, 776)], np.int32),
            "DOOR_CORNER_ROI": np.array([[423, 33], [420, 11], [442, 3], [620, 1]], np.int32),
            "STANDING_ZONE": np.array([(595, 892), (644, 996), (733, 950), (699, 814), (575, 858)], np.int32),
            "INTERACTION_ZONE": np.array([(7, 2), (2, 1272), (2085, 1009), (2114, 4)], np.int32)
        }
    }
]

BASE_OUTPUT_DIR = "strong_room_opening"

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
        "grace_buffer_a": 0,
        "grace_buffer_b": 0,
        "improper_positioning": None,
        "violation_type": None,
    }

def calculate_min_unlock_frames(fps):
    """Convert minimum unlock seconds to frame count based on actual video FPS."""
    return int(MIN_UNLOCK_SECONDS * fps)

def calculate_max_unlock_frames(fps):
    """Convert maximum unlock seconds to frame count based on actual video FPS."""
    return int(MAX_UNLOCK_SECONDS * fps)
