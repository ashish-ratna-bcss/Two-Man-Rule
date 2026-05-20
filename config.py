# config.py
import numpy as np

# ============ VIDEO & FPS ============
DEFAULT_FPS = 30  # Will be overridden by actual video FPS
GRACE_BUFFER_FRAMES = 15  # 0.5s at 30 FPS

EVENING_SECOND_UNLOCKER_TIMEOUT_SECONDS = 300.0  # fallback default; prefer per-stream "evening_second_unlocker_timeout_seconds"

# ============ TIMERS (in seconds) ============
# One valid key-turn/unlock interaction must last at least 5s and no more than 10s.
MIN_UNLOCK_SECONDS = 5.0
MAX_UNLOCK_SECONDS = 10.0

# ============ GEOMETRIC CONSTRAINTS ============
HAND_LOCK_PROXIMITY_PIXELS = 80
ELBOW_LOCK_PROXIMITY_PIXELS = 130  # Elbows can be offset when the unlocker faces away
WRIST_CONFIDENCE_THRESHOLD = 0.5  # Fallback if wrist < 0.5
ARM_KEYPOINT_CONFIDENCE_THRESHOLD = 0.25
HEAD_CONFIDENCE_THRESHOLD = 0.25
DOOR_FACING_ARM_RAISE_PIXELS = 30
LEFT_RIGHT_ORDER_MIN_PIXELS = 25
UNLOCKER_ANCHOR_MATCH_PIXELS = 95
MAX_SYNTHETIC_HOLD_FRAMES = 60  # ~2.0s at 30 FPS. Dropped after this if not re-detected.
DEPARTURE_FRAMES_THRESHOLD = 15  # ~0.5s at 30 FPS. Frames absent from tracker before verified unlocker marked departed.

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
# Default debounce (frames) required to accept a door state change
DOOR_DEBOUNCE_FRAMES = 20
# Global toggle to enable special darkening (lights-off) protection
DOOR_DARKENING_PROTECTION = True
# Minimum visible area ratio required before using occlusion-aware door verification
DOOR_CORNER_MIN_VISIBLE_RATIO = 0.5

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
# Optimization & Production Flags
RTSP_LOW_LATENCY = True
STAGGER_START_DELAY = 2.0  # Seconds between stream launches
MAX_PROCESS_VRAM_FRACTION = None  # Optional: e.g. 0.3 to limit each process


# ============ SHARED INFERENCE & SCALING ============
# When True, multiple streams share a single YOLO instance to save VRAM.
SHARED_INFERENCE_ENABLED = True   # Enabled by default for high-density scalability

# Scanning server: handles all idle/background streams in a larger batch.
BATCH_SIZE_LIMIT = 16             # Reduced from 32 — lower latency per batch
INFERENCE_BATCH_WAIT_MS = 5       # Max wait time (ms) to collect a batch

# Priority server: handles active-unlocker streams with a small, fast batch.
PRIORITY_BATCH_SIZE_LIMIT = 4     # Small batch → fast GPU turnaround for active streams

GPU_IDLE_TIMEOUT = 300            # Seconds to keep model in VRAM after last use
MAX_SHARED_MEMORY_MB = 4096       # Buffer for passing frames between processes (4GB)

# Pre-warm the GPU model on server startup (before first request arrives).
# Eliminates cold-start latency at window open. Set False only for RAM-constrained hosts.
GPU_PRELOAD_ON_START = True

# ============ INFERENCE TIMEOUT & LKG (Last-Known-Good) ============
# SCANNING tier: how long a camera worker waits before treating a GPU reply as lost.
# Set to 3× your p99 latency measured from gpu_server_scan.log.
INFERENCE_TIMEOUT_SECONDS: float = 2.0

# PRIORITY tier: tighter timeout for active-unlocker streams.
# START at 2.0s (same as scanning) — tune DOWN to p99_priority_ms * 2 / 1000
# once gpu_server_priority.log confirms real GPU latency.
# Example: if priority p99 = 80ms → set to 0.16s
PRIORITY_INFERENCE_TIMEOUT_SECONDS: float = 2.0

# How many consecutive inference timeouts to tolerate before the tracker stops
# coasting on LKG and starts using empty detections (track ageing).
# During LKG reuse the FSM is frozen — timers don't reset, UNAUTHORIZED is not fired.
# After this threshold the camera logs DEGRADED (ops alert) instead of UNAUTHORIZED.
# Formula: LKG_MAX_CONSECUTIVE_TIMEOUTS × process_every / fps = coast duration seconds
# Default: 15 × 3 / 15 = 3.0s of coast time before graceful track ageing begins.
LKG_MAX_CONSECUTIVE_TIMEOUTS: int = 15


STREAMS_CONFIG = [
    # Stream 0
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.108.159:8001/Streaming/Channels/4001",
        "camera_id": "GF-1-CAM-40",
        "site_id": "1",
        "site_name": "somajiguda",
        "closed_door_reference": "close_doors/closed_GF-1-CAM-40.jpg",
        "ssim_threshold": 0.80,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 2.5,
        "max_unlock_seconds": 10.0,

        "evening_second_unlocker_timeout_seconds": 300.0,
        "mirror_left_right": False,
        "rois": {
            "LOCKS_ROI": np.array([(584, 272), (888, 117), (1002, 525), (673, 690)], np.int32),
            "DOOR_ROI": np.array([(548, 55), (787, 721), (1047, 570), (926, 2), (665, 0)], np.int32),
            "DOOR_CORNER_ROI": np.array([[640.30, 13.22], [659.51, 73.73], [845.86, 9.38]], np.int32),
            "STANDING_ZONE": np.array([(801, 753), (1067, 597), (1184, 724), (882, 887)], np.int32),
            "INTERACTION_ZONE": np.array([(272, 160), (1857, 2), (1750, 714), (275, 1392)], np.int32)
        }
    },
    # Stream 1
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.108.159:8002/Streaming/Channels/4201",
        "camera_id": "FF-1-CAM-42",
        "site_id": "1",
        "site_name": "somajiguda",
        "closed_door_reference": "close_doors/closed_FF-1-CAM-42.jpg",
        "ssim_threshold": 0.80,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 2.5,
        "max_unlock_seconds": 10.0,

        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "LOCKS_ROI": np.array([(333, 247), (809, 150), (876, 558), (433, 683)], np.int32),
            "DOOR_ROI": np.array([(401, 1), (817, 0), (891, 638), (548, 776)], np.int32),
            "DOOR_CORNER_ROI": np.array([[423, 33], [420, 11], [442, 3], [620, 1]], np.int32),
            "STANDING_ZONE": np.array([(595, 892), (644, 996), (733, 950), (699, 814), (575, 858)], np.int32),
            "INTERACTION_ZONE": np.array([(7, 2), (2, 1272), (2085, 1009), (2114, 4)], np.int32)
        }
    },
    # Stream 2
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.121.130:8001/Streaming/Channels/2101",
        "camera_id": "GF-2-CAM-21",
        "site_id": "2",
        "site_name": "jubilee_hills",
        "closed_door_reference": "close_doors/closed_GF-2-CAM-21.jpg",
        "ssim_threshold": 0.80,
        "debounce_threshold": 20,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 5.0,
        "max_unlock_seconds": 10.0,

        "evening_second_unlocker_timeout_seconds": 50.0,
        "rois": {
            "LOCKS_ROI": np.array([(552, 475), (1212, 240), (1240, 909), (669, 1163)], np.int32),
            "DOOR_ROI": np.array([(330, 113), (634, 0), (1217, 0), (1221, 877), (615, 1168)], np.int32),
            "DOOR_CORNER_ROI": np.array([[488, 113], [493, 172], [609, 72]], np.int32),
            "STANDING_ZONE": np.array([[688, 1178], [1227, 918], [1420, 1118], [929, 1416]], np.int32),
            "INTERACTION_ZONE": np.array([[6, 11], [2615, 0], [1927, 1511], [25, 1508]], np.int32)
        }
    },
    # Stream 3
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.76.17:8001/Streaming/Channels/2001",
        "camera_id": "GF-4-CAM-20",
        "site_id": "4",
        "site_name": "vijayawada",
        "closed_door_reference": "close_doors/closed_GF-4-CAM-20.jpg",
        "ssim_threshold": 0.80,
        "debounce_threshold": 5,
        "intensity_threshold": 10,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.3,
        "min_unlock_seconds": 1.0,
        "max_unlock_seconds": 10.0,

        "evening_second_unlocker_timeout_seconds": 50.0,
        "rois": {
            "STANDING_ZONE": np.array([(822, 792), (958, 989), (1423, 735), (1264, 570)], np.int32),
            "DOOR_ROI": np.array([(662, 4), (780, 753), (1333, 501), (1319, 0)], np.int32),
            "LOCKS_ROI": np.array([(708, 31), (1217, 31), (1224, 425), (507, 478)], np.int32),
            "DOOR_CORNER_ROI": np.array([(662, 4), (780, 753), (1333, 501), (1319, 0)], np.int32),
            "INTERACTION_ZONE": np.array([(13, 11), (10, 1393), (2675, 157), (2679, 8)], np.int32)
        }
    },
    # Stream 4
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@106.51.37.109:8001/Streaming/Channels/2501",
        "camera_id": "GF-5-CAM-25",
        "site_id": "5",
        "site_name": "jayanagar",
        "closed_door_reference": "close_doors/closed_GF-5-CAM-25.jpg",
        "ssim_threshold": 0.80,
        "debounce_threshold": 15,
        "intensity_threshold": 8,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 4.0,
        "max_unlock_seconds": 10.0,

        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "DOOR_CORNER_ROI": np.array([(803, 2), (908, 1), (798, 37)], np.int32),
            "DOOR_ROI": np.array([(581, 3), (1187, 0), (1034, 742), (471, 797)], np.int32),
            "LOCKS_ROI": np.array([(526, 81), (1144, 49), (1106, 557), (501, 588)], np.int32),
            "STANDING_ZONE": np.array([(545, 897), (544, 1103), (1165, 1050), (1154, 853)], np.int32),
            "INTERACTION_ZONE": np.array([(0, 8), (0, 1340), (1902, 1416), (2076, 11)], np.int32)
        }
    },
    # Stream 5
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@106.51.52.103:8002/Streaming/Channels/2101",
        "camera_id": "FF-3-CAM-21",
        "site_id": "3",
        "site_name": "vizag",
        "closed_door_reference": "close_doors/closed_FF-3-CAM-21.jpg",
        "ssim_threshold": 0.70,
        "debounce_threshold": 7,
        "intensity_threshold": 12,
        "motion_threshold": 5.0,
        "door_corner_min_visible_ratio": 0.3,
        # NOTE: min_unlock_seconds was 0.2 (200ms) — almost certainly a calibration
        # accident.  At process_every=3 and 15fps that is ~3 frames: one inference
        # cycle.  The camera had 160 inference timeouts and zero authorized captures
        # in the incident logs.  Raised to 1.5s (a conservative minimum) so the
        # timer has time to accumulate across multiple successful inference cycles.
        # Re-tune upward (2.5–3.0s recommended) once the queue fix is deployed
        # and you can confirm clean detections on this stream.
        "min_unlock_seconds": 1.0,
        "max_unlock_seconds": 10.0,

        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "STANDING_ZONE": np.array([(1116, 411), (983, 528), (1293, 664), (1569, 607)], np.int32),
            "LOCKS_ROI": np.array([(955, 2), (1046, 251), (1523, 192), (1523, 192), (1537, 2)], np.int32),
            "DOOR_ROI": np.array([(1091, 5), (1110, 373), (1671, 607), (1737, 11)], np.int32),
            "DOOR_CORNER_ROI": np.array([(1091, 5), (1110, 373), (1671, 607), (1737, 11)], np.int32),
            "INTERACTION_ZONE": np.array([(3, 18), (0, 1172), (2675, 541), (2679, 8)], np.int32)
        }
    },
    # Stream 6
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.97.83:8001/Streaming/Channels/1801",
        "camera_id": "GF-6-CAM-18",
        "site_id": "6",
        "site_name": "himayatnagar",
        "closed_door_reference": "close_doors/closed_GF-6-CAM-18.jpg",
        "ssim_threshold": 0.80,
        "debounce_threshold": 20,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 3.5,
        "max_unlock_seconds": 10.0,

        "evening_second_unlocker_timeout_seconds": 50.0,
        "rois": {
            "STANDING_ZONE": np.array([[862, 931], [1087, 687], [1217, 728], [1217, 728], [1027, 1013]], np.int32),
            "LOCKS_ROI": np.array([(659, 284), (1065, 107), (1141, 642), (799, 861)], np.int32),
            "DOOR_ROI": np.array([[571, 5], [761, 969], [1116, 620], [1040, 0]], np.int32),
            "DOOR_CORNER_ROI": np.array([[591, 16], [613, 38], [811, 41], [814, 4]], np.int32),
            "INTERACTION_ZONE": np.array([[0, 8], [10, 1511], [2679, 1511], [2688, 24]], np.int32)
        }
    },
    # Stream 7
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.106.41:8001/Streaming/Channels/2901",
        "camera_id": "GF-10-CAM-29",
        "site_id": "10",
        "site_name": "kompally",
        "closed_door_reference": "close_doors/closed_GF-10-CAM-29.jpg",
        "ssim_threshold": 0.80,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 4.0,
        "max_unlock_seconds": 10.0,

        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "DOOR_CORNER_ROI": np.array([(2342, 106), (2197, 81), (2082, 24), (2070, 47), (2310, 165)], np.int32),
            "LOCKS_ROI": np.array([(1717, 106), (1608, 393), (2131, 631), (2240, 294)], np.int32),
            "STANDING_ZONE": np.array([[1587, 576], [2035, 815], [1947, 980], [1480, 716]], np.int32),
            "DOOR_ROI": np.array([[1750, 5], [1563, 550], [2095, 801], [2460, 5]], np.int32),
            "INTERACTION_ZONE": np.array([[0, 5], [3, 224], [1369, 1520], [2685, 1511], [2688, 5]], np.int32)
        }
    }
]

BASE_OUTPUT_DIR = "strong_room_opening"
BASE_LOG_DIR = "logs"


def create_session():
    """Create a fresh session state dict.

    All keys that main.py or state_machine.py ever reads from session must be
    initialised here.  Missing keys cause silent KeyErrors when reset_session()
    is called mid-stream — a particularly hard class of bug because the error
    only surfaces in the second auth window of a 24-hour run.
    """
    return {
        # ---- FSM sequence state ----
        "sequence_state": "WAITING_FOR_FIRST_UNLOCKER",

        # ---- Candidates: person currently attempting an unlock (not yet verified) ----
        "candidate_a": None,
        "candidate_b": None,

        # ---- Official IDs: assigned ONLY after one complete lock interaction ----
        "id_a": None,
        "id_b": None,

        # ---- Timers ----
        "timer_a_frames": 0,
        "timer_b_frames": 0,
        "timer_a_seconds": 0.0,
        "timer_b_seconds": 0.0,

        # ---- Grace buffers (frames a verified unlocker can be absent before departure) ----
        "grace_buffer_a": 0,
        "grace_buffer_b": 0,

        # ---- Violation tracking ----
        "improper_positioning": None,
        "violation_type": None,
        "captured_violations": [],
        "same_id_return_timer_frames": 0,
        "same_id_return_grace_frames": 0,

        # ---- Door state tracking ----
        # Set True once a DOOR_OPEN_* capture has been written for this session,
        # so main.py does not fire a second capture on the same opening event.
        "door_open_captured": False,
        # Frame index when the CLOSED→OPEN transition was first detected.
        # Reset to None when door closes.  Used by morning post-open grace window.
        "door_opening_start_frame": None,
        # Frame index when the OPEN→CLOSED transition was first detected.
        # Reset to None when door opens.  Used by evening countdown timer.
        "door_closing_start_frame": None,
    }


def calculate_min_unlock_frames(fps):
    """Convert minimum unlock seconds to frame count based on actual video FPS."""
    return int(MIN_UNLOCK_SECONDS * fps)


def calculate_max_unlock_frames(fps):
    """Convert maximum unlock seconds to frame count based on actual video FPS."""
    return int(MAX_UNLOCK_SECONDS * fps)