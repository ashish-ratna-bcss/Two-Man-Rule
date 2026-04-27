# config.py

# ============ VIDEO & FPS ============
DEFAULT_FPS = 30  # Will be overridden by actual video FPS
GRACE_BUFFER_FRAMES = 15  # 0.5s at 30 FPS

# ============ TIMERS (in seconds) ============
DWELL_THRESHOLD_SECONDS = 10.0

# ============ GEOMETRIC CONSTRAINTS ============
ELBOW_LOCK_PROXIMITY_PIXELS = 40  # "Close enough" for kinematic fallback
WRIST_CONFIDENCE_THRESHOLD = 0.5  # Fallback if wrist < 0.5

# ============ POSE KEYPOINT INDICES (YOLOv11) ============
# Standard COCO format: 0=Nose, 1=L_Eye, 2=R_Eye, ..., 9=L_Wrist, 10=R_Wrist, etc.
KEYPOINT_WRIST_LEFT = 9
KEYPOINT_WRIST_RIGHT = 10
KEYPOINT_ELBOW_LEFT = 7
KEYPOINT_ELBOW_RIGHT = 8
KEYPOINT_SHOULDER_LEFT = 5
KEYPOINT_SHOULDER_RIGHT = 6

# ============ ROI DEFINITIONS (PLACEHOLDERS - user provides) ============
LOCK_A_ROI = None
LOCK_B_ROI = None
DOOR_ROI = None
INTERACTION_ZONE = None

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

# ============ BYTETRACK PARAMETERS ============
TRACK_BUFFER = 30
TRACK_THRESH = 0.5

# ============ MODEL PATHS ============
YOLO_POSE_MODEL = "yolov11-pose.pt"
CLOSED_DOOR_REFERENCE = "assets/closed_ref.jpg"

def create_session():
    """Create a fresh session state dict."""
    return {
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
    }

def calculate_dwell_frames(fps):
    """Convert 10 seconds to frame count based on actual video FPS."""
    return int(DWELL_THRESHOLD_SECONDS * fps)
