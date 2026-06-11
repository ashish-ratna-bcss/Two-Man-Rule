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
# SSIM margin by which entering OPEN is made stricter than returning to CLOSED
# (hysteresis). Biases the door verifier toward CLOSED so a brief/marginal SSIM
# dip cannot false-open. Per-stream override: "door_open_hysteresis".
DOOR_OPEN_HYSTERESIS = 0.05
# Minimum visible area ratio required before using occlusion-aware door verification
DOOR_CORNER_MIN_VISIBLE_RATIO = 0.5

# ---- Bright-washout structural veto (Fix 1) ----
# When the door ROI brightness has shifted far from the reference (dawn sunlight /
# overexposure), raw-grayscale SSIM collapses even though the door is shut. In that
# regime we additionally require a GRADIENT (structural) drop before reading OPEN: a
# pure lighting wash leaves edges intact (high grad-SSIM) and is vetoed, while a real
# opening destroys the door-edge structure (low grad-SSIM) and still opens. Below the
# intensity delta the door logic is unchanged (no regression on normal-light opens).
DOOR_WASHOUT_INTENSITY_DELTA = 25.0   # |curr_mean - ref_mean| above which veto is active
DOOR_GRAD_SSIM_OPEN_MAX = 0.85        # grad-SSIM must be below this to confirm a real open

# ---- Same-person appearance gate (Fix 5) — DISABLED ----
# A normalized HSV color histogram cannot work here: all staff wear the SAME uniform,
# so two DIFFERENT people score as highly similar → the gate would false-flag a legit
# second unlocker as SAME_ID and block real dual-auth. Multi-angle cameras make a flat
# color descriptor worse still. Superseded by a deep Re-ID tracker (BoT-SORT) that
# keeps a stable identity per person through occlusion/re-entry, so the existing
# excluded-id SAME_ID logic catches the same body without a color match. Left wired but
# OFF; do not enable in a same-uniform deployment.
APPEARANCE_GATE_ENABLED = False
DOOR_SAME_PERSON_APPEARANCE_SIM = 0.90

# ============ ALERT SYSTEM ============

# ============ OCCLUSION FALLBACK ============
ANKLE_OCCLUSION_CONFIDENCE_THRESHOLD = 0.3  # Both ankles below this = fallback mode
HEAD_TO_LOCKS_MAX_PIXELS = 150  # Max distance from head to LOCKS_ROI polygon

# ============ BYTETRACK PARAMETERS ============
TRACK_BUFFER = 30
TRACK_THRESH = 0.5

# ---- Deep appearance Re-ID (Option A: augment the pose-keypoint matcher) ----
# A deep OSNet embedding (build/face/gait/texture, trained cross-view) re-links a
# person across camera angle and long gaps where pose-keypoint matching fails — the
# fix for same-uniform, multi-angle id-switches. When the model is unavailable the
# tracker falls back to the existing pose-keypoint Re-ID, so nothing breaks if the
# weights file is absent or the kill-switch is off.
REID_APPEARANCE_ENABLED = True
REID_MODEL_PATH = "models/weights/osnet_x0_25_msmt17.onnx"
# Cosine-distance ceiling for accepting a re-link. Higher = more aggressive re-link
# (catches one-person-does-both-unlocks, the security-critical miss) at some risk of
# merging two similar staff. Tune against real footage.
REID_COSINE_MAX = 0.40
REID_GALLERY_EMA = 0.9   # EMA weight for the per-track embedding template

# ============ MODEL PATHS ============
YOLO_POSE_MODEL = "yolov8n-pose.pt"  # Lightweight nano model, ~6.5MB

# ============ STREAMS & ORCHESTRATION CONFIG ============
# Optimization & Production Flags
RTSP_INGEST_BACKEND = "pyav"
RTSP_TRANSPORT = "tcp"
RTSP_JITTER_LATENCY_MS = 1000
RTSP_DROP_ON_LATENCY = False
RTSP_READ_TIMEOUT_SECONDS = 1.5
RTSP_STARTUP_TIMEOUT_SECONDS = 10.0
# Initial-connect retry: a transient first-open failure (camera/network down at dawn)
# retries in-process with backoff instead of killing the worker (witnessed: ongole 22
# crash-loop restarts on [Errno 110]). reconnect_delay (5s) × 12 ≈ 1 min of riding a blip;
# a genuinely dead source still raises after the budget for the supervisor to handle.
RTSP_INITIAL_OPEN_MAX_ATTEMPTS = 12
RTSP_LOW_LATENCY = False
PRESERVE_FILE_FRAMES = True       # Offline videos are read sequentially with no frame overwrite/drop.
RTSP_PREFER_REALTIME = True       # Live RTSP keeps latest frame to avoid latency buildup.
STAGGER_START_DELAY = 2.0  # Seconds between stream launches
# Cap each child process to a fraction of total VRAM so one leaky stream
# cannot OOM siblings. YOLOv8n-pose @ FP16 ~720 MiB; 0.08 × 32 GB = ~2.5 GB
# leaves 3× headroom for activation spikes.
MAX_PROCESS_VRAM_FRACTION = 0.08


# ============ PRESENCE DETECTION (morning window lazy-trigger) ============
# Background subtractor (MOG2) warmup in seconds — converted to frames using actual stream FPS
PRESENCE_WARMUP_SECONDS: float = 4.0   # seconds; frame count = PRESENCE_WARMUP_SECONDS * fps
# Foreground pixel count threshold on the INTERACTION_ZONE crop to confirm presence
PRESENCE_PIXEL_THRESHOLD: int = 2500   # tune per deployment if needed

# ============ INFERENCE TIMEOUT & LKG (Last-Known-Good) ============
# How long to wait before treating an inference as lost (timeout).
INFERENCE_TIMEOUT_SECONDS: float = 8.0

# How many consecutive inference timeouts to tolerate before the tracker stops
# coasting on LKG and starts using empty detections (track ageing).
# During LKG reuse the FSM is frozen — timers don't reset, UNAUTHORIZED is not fired.
# After this threshold the camera logs DEGRADED (ops alert) instead of UNAUTHORIZED.
# Formula: LKG_MAX_CONSECUTIVE_TIMEOUTS × process_every / fps = coast duration seconds
# Default: 15 × 3 / 15 = 3.0s of coast time before graceful track ageing begins.
LKG_MAX_CONSECUTIVE_TIMEOUTS: int = 15


# ============ FRAME QUALITY / VIDEO DEGRADATION ============
FRAME_QUALITY_ENABLED = True
FRAME_QUALITY_DEGRADED_AFTER_FRAMES = 15
FRAME_QUALITY_RECOVERY_GOOD_FRAMES = 20
FRAME_QUALITY_STALE_AFTER_FRAMES = 90

# Localized-corruption (decode block) detection. Global white/black ratios miss a
# blowout confined to one quadrant + macroblock smear (witnessed: vizag/rajahmundry/
# marathahalli June-8 dawn false-opens, gate logged 0 freezes). A cell-grid catches a
# saturated block in any single cell. Empirically separated on the real corrupt vs clean
# captures: corrupt cell-white 0.92-1.0, flat-white frac 0.05-0.13; clean cell-white
# <=0.21, flat-white 0.0. Thresholds biased toward the clean side for low false-positives.
FRAME_QUALITY_CELL_GRID = (8, 6)            # (cols, rows) cells over the small frame
FRAME_QUALITY_CELL_WHITE_RATIO = 0.85       # any cell >= this fraction saturated -> corrupt
FRAME_QUALITY_FLAT_WHITE_FRACTION = 0.03    # frac of frame that is flat (decode) white -> corrupt

# Adaptive recovery: a perpetually freeze-storming stream (witnessed: kompally 5080 freeze
# / 65 recover) can never accumulate RECOVERY_GOOD_FRAMES consecutive clean frames, so it
# stays frozen the whole window. After a long bad run, lower the bar to a floor so the
# stream resumes real-time analysis. Each counted frame already passed every corruption
# check, so a smaller count is still all-clean.
FRAME_QUALITY_RECOVERY_GOOD_FRAMES_MIN = 8  # floor when storming
FRAME_QUALITY_RECOVERY_STORM_BAD_FRAMES = 60  # bad-run length that triggers the lowered floor

# Morning capture grace: the CLOSED->OPEN snapshot previously fired at the exact transition
# instant, before unlocker IDs settled -> p1/p2 null even when openers were present
# (witnessed: kakinada/himayatnagar/nizamabad June-8). After an unauthorized-looking
# transition, hold this many real seconds of GOOD frames, re-checking auth, so the snapshot
# reflects the settled state. Authorized + SAME_ID verdicts still fire immediately.
MORNING_CAPTURE_GRACE_SECONDS = 2.0


STREAMS_CONFIG = [
    #Telangana streams Below

    # Stream 0
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.108.159:8001/Streaming/Channels/4001",
        "camera_id": "GF-1-CAM-40",
        "site_id": "1",
        "site_name": "somajiguda",
        "closed_door_reference": "close_doors/closed_GF-1-CAM-40.jpg",
        "ssim_threshold": 0.9,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 2.5,
        "max_unlock_seconds": 10.0,

        "evening_second_unlocker_timeout_seconds": 300.0,
        "mirror_left_right": False,
        "rois": {
            "LOCKS_ROI": np.array([(584.0115107913666, 312.76661772144774), (984.9553956834528, 146.45870405238304), (1055.8618705035967, 472.6284882250448), (716.7999999999997, 633.7795673617353)], np.int32),
            "DOOR_ROI": np.array([(548, 55), (787, 721), (1047, 570), (926, 2), (665, 0)], np.int32),
            "DOOR_CORNER_ROI": np.array([[640.30, 13.22], [659.51, 73.73], [845.86, 9.38]], np.int32),
            "STANDING_ZONE": np.array([(801, 753), (1067, 597), (1184, 724), (882, 887)], np.int32),
            "INTERACTION_ZONE": np.array([(272, 160), (1857, 2), (1750, 714), (275, 1392)], np.int32)
        }
    },
    # Disabled Stream - Somajiguda FF
    # {
    #     "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.108.159:8002/Streaming/Channels/4201",
    #     "camera_id": "FF-1-CAM-42",
    #     "site_id": "1",
    #     "site_name": "somajiguda",
    #     "closed_door_reference": "close_doors/closed_FF-1-CAM-42.jpg",
    #     "ssim_threshold": 0.80,
    #     "debounce_threshold": 15,
    #     "intensity_threshold": 6,
    #     "motion_threshold": 3.0,
    #     "door_corner_min_visible_ratio": 0.5,
    #     "min_unlock_seconds": 2.5,
    #     "max_unlock_seconds": 10.0,

    #     "evening_second_unlocker_timeout_seconds": 300.0,
    #     "rois": {
    #         "LOCKS_ROI": np.array([(333, 247), (809, 150), (876, 558), (433, 683)], np.int32),
    #         "DOOR_ROI": np.array([(401, 1), (817, 0), (891, 638), (548, 776)], np.int32),
    #         "DOOR_CORNER_ROI": np.array([[423, 33], [420, 11], [442, 3], [620, 1]], np.int32),
    #         "STANDING_ZONE": np.array([(595, 892), (644, 996), (733, 950), (699, 814), (575, 858)], np.int32),
    #         "INTERACTION_ZONE": np.array([(7, 2), (2, 1272), (2085, 1009), (2114, 4)], np.int32)
    #     }
    # },
    # # Disabled Stream - additional Somajiguda stream
    # {
    #     "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.108.159:8004/Streaming/Channels/301",
    #     "camera_id": "TF-1-CAM-03",
    #     "site_id": "1",
    #     "site_name": "somajiguda",
    #     "closed_door_reference": "close_doors/closed_TF-1-CAM-03.jpg",
    #     "ssim_threshold": 0.80,
    #     "debounce_threshold": 20,
    #     "intensity_threshold": 6,
    #     "motion_threshold": 3.0,
    #     "door_corner_min_visible_ratio": 0.5,
    #     "min_unlock_seconds": 5.0,
    #     "max_unlock_seconds": 10.0,

    #     "evening_second_unlocker_timeout_seconds": 300.0,
    #     "rois": {
    #         "STANDING_ZONE": np.array([(2695.7683372532724, 449.2897989300864), (2689.3931009038906, 550.1125226621584), (2518.2069728832303, 509.9723271262794), (2530.012912746724, 404.89946234118446)], np.int32),
    #         "LOCKS_ROI": np.array([(2427.301235934328, 51.42962553034492), (2744.961941161355, 90.19527235354568), (2712.0739658273365, 384.8715313463512), (2395.425198302895, 326.50802434975077)], np.int32),
    #         "DOOR_ROI": np.array([(2755.4860932682404, 6.002055498458372), (2693.6566996402858, 504.5837615621785), (2355.568313206576, 417.7595066803697), (2402.9269976875626, 4.6865364850976325)], np.int32),
    #         "DOOR_CORNER_ROI": np.array([(2509.4705485611494, 3.0252720900202896), (2576.7644057830644, 4.205866076369671), (2507.1093605884507, 34.901309721453586)], np.int32),
    #         "INTERACTION_ZONE": np.array([(15.347721822541956, 11.289429994465959), (3830.520491694877, 11.595429538721955), (3804.3462345508183, 2130.4556354916053), (41.43884892086331, 2129.7841726618703)], np.int32)
    #     }
    # },
    # Stream 1
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.121.130:8001/Streaming/Channels/2101",
        "camera_id": "GF-2-CAM-21",
        "site_id": "2",
        "site_name": "jubilee_hills",
        "closed_door_reference": "close_doors/closed_GF-2-CAM-21.jpg",
        "ssim_threshold": 0.90,
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
            "DOOR_CORNER_ROI": np.array([[492.70910435113416, 115.53587898911634], [687.1144595739972, 55.446540880503115], [703.3660319010413, 76.4779874213836], [581.9572268695948, 126.18867924528294], [528.4226356746262, 175.89937106918228], [511.71666753135923, 241.15107913669053]], np.int32),
            "STANDING_ZONE": np.array([[688, 1178], [1227, 918], [1420, 1118], [929, 1416]], np.int32),
            "INTERACTION_ZONE": np.array([[6, 11], [2615, 0], [1927, 1511], [25, 1508]], np.int32)
        }
    },
    # Stream 2
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.97.83:8001/Streaming/Channels/1801",
        "camera_id": "GF-6-CAM-18",
        "site_id": "6",
        "site_name": "himayatnagar",
        "closed_door_reference": "close_doors/closed_GF-6-CAM-18.jpg",
        "ssim_threshold": 0.892,
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
    # Stream 3
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.106.41:8001/Streaming/Channels/2901",
        "camera_id": "GF-10-CAM-29",
        "site_id": "10",
        "site_name": "kompally",
        "closed_door_reference": "close_doors/closed_GF-10-CAM-29.jpg",
        "ssim_threshold": 0.684,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 4.0,
        "max_unlock_seconds": 10.0,

        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "DOOR_CORNER_ROI": np.array([[1859.104942007931, 5.622578859988942], [1839.5516852120613, 96.82309304950807], [1897.768101919044, 41.670698274471825], [2000.719238832445, 41.05789388808254], [2002.074873754841, 4.796163069544375]], np.int32),
            "LOCKS_ROI": np.array([(1717, 106), (1608, 393), (2131, 631), (2240, 294)], np.int32),
            "STANDING_ZONE": np.array([[1587, 576], [2035, 815], [1947, 980], [1480, 716]], np.int32),
            "DOOR_ROI": np.array([[1750, 5], [1563, 550], [2095, 801], [2460, 5]], np.int32),
            "INTERACTION_ZONE": np.array([[0, 5], [3, 224], [1369, 1520], [2685, 1511], [2688, 5]], np.int32)
        }
    },
    # Disabled Stream - additional Jubilee Hills stream
    # {
    #     "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.121.130:8002/Streaming/Channels/1101",
    #     "camera_id": "FF-2-CAM-11",
    #     "site_id": "2",
    #     "site_name": "jubilee_hills",
    #     "closed_door_reference": "close_doors/closed_FF-2-CAM-11.jpg",
    #     "ssim_threshold": 0.80,
    #     "debounce_threshold": 20,
    #     "intensity_threshold": 6,
    #     "motion_threshold": 3.0,
    #     "door_corner_min_visible_ratio": 0.5,
    #     "min_unlock_seconds": 5.0,
    #     "max_unlock_seconds": 10.0,

    #     "evening_second_unlocker_timeout_seconds": 50.0,
    #     "rois": {
    #         "STANDING_ZONE": np.array(
    #             [(545.8414513929836, 740.9811320754717), (1106.8980551665684, 572.9811320754717),
    #              (1279.5395683453237, 821.2374100719425), (667.1654676258993, 1040.4028776978419)],
    #             np.int32
    #         ),
    #         "DOOR_ROI": np.array(
    #             [(513.2535807291663, 6.609686609686621), (1219.7150997150989, 5.014245014245027),
    #              (1402.3931623931621, 139.4871794871795), (1373.6752136752134, 917.2649572649572),
    #              (516.4444444444445, 1108.4444444444446)],
    #             np.int32
    #         ),
    #         "LOCKS_ROI": np.array(
    #             [(512.278068529673, 16.720451293957353), (1292.6394786215064, 11.411870272924473),
    #              (1199.739310753431, 799.7361518963072), (531.9259259259259, 852.8888888888889)],
    #             np.int32
    #         ),
    #         "DOOR_CORNER_ROI": np.array(
    #             [(562.8413531510314, 57.5085195001893), (667.1654728026784, 54.964028776978395),
    #              (560.2968624278205, 89.73873532752741)],
    #             np.int32
    #         ),
    #         "INTERACTION_ZONE": np.array(
    #             [(45.388367729831145, 48.91557223264539), (2634.5925925925926, 30.962962962962962),
    #              (2637.5684803001877, 1481.1707317073171), (40.34521575984991, 1460.9981238273922)],
    #             np.int32
    #         )
    #     }
    # }
    # ,
    # Stream 4 - Kokapet
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.112.217:8001/Streaming/Channels/1501",
        "camera_id": "GF-12-CAM-15",
        "site_id": "12",
        "site_name": "kokapet",
        "closed_door_reference": "close_doors/closed_GF-12-CAM-15.jpg",
        # PROVISIONAL 2026-06-10: measured evening closed-door SSIM = 0.62 (ref image
        # is bright mean~80, evening is dark mean~45 → reference lighting mismatch), yet
        # old threshold 0.811 sat ABOVE it → closed door leaned OPEN (false-open risk).
        # Lowered below the evening closed band (active ~0.52 after twilight relax →
        # closed margin +0.10). Single-frame estimate; confirm/refine from VIDEO_FRAME
        # logs. Better long-term fix: an evening-lit closed_door_reference.
        "ssim_threshold": 0.55,
        "door_open_hysteresis": 0.06,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 5.0,
        "max_unlock_seconds": 10.0,

        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "DOOR_ROI": np.array([(1082, 106), (1518, 14), (2123, 78), (1733, 1445), (1174, 1362)], np.int32),
            "LOCKS_ROI": np.array([(1144, 459), (1192, 1210), (1582, 1443), (1842, 1007), (1991, 336)], np.int32),
            "STANDING_ZONE": np.array([(1099, 1436), (1503, 790), (1886, 931), (1679, 1501)], np.int32),
            "INTERACTION_ZONE": np.array([(2480, 37), (2584, 1481), (170, 1463), (238, 46)], np.int32),
            "DOOR_CORNER_ROI": np.array([(1129, 158), (1262, 56), (1269, 120), (1132, 246)], np.int32)
        }
    }
        ,
        # Stream 5 - Khammam
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.113.18:8001/Streaming/Channels/3101",
        "camera_id": "GF-30-CAM-31",
        "site_id": "30",
        "site_name": "khammam",
        "closed_door_reference": "close_doors/closed_GF-30-CAM-31.jpg",
        "ssim_threshold": 0.786,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 5.0,
        "max_unlock_seconds": 10.0,

        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "INTERACTION_ZONE": np.array([(26, 48), (2646, 35), (2659, 1485), (32, 1485)], np.int32),
            "STANDING_ZONE": np.array([(444, 1153), (1266, 962), (1452, 1370), (550, 1511)], np.int32),
            "LOCKS_ROI": np.array([(424, 1300), (548, 256), (1768, 266), (1678, 1347)], np.int32),
            "DOOR_CORNER_ROI": np.array([(798.5931254996001, 64.10871302957635), (855.8912869704234, 60.52757793764989), (864.8441247002395, 10.391686650679482), (1036.7386091127096, 10.391686650679482), (743.08553157474, 207.35411670663467)], np.int32),
            "DOOR_ROI": np.array([(621, 20), (1899, 23), (1641, 1334), (413, 1240)], np.int32)
        }
    }
    ,
    # Stream 6 - Nizamabad
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@43.249.216.149:8001/Streaming/Channels/1601",
        "camera_id": "GF-31-CAM-16",
        "site_id": "31",
        "site_name": "nizamabad",
        "closed_door_reference": "close_doors/closed_GF-31-CAM-16.jpg",
        "ssim_threshold": 0.833,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 5.0,
        "max_unlock_seconds": 10.0,

        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "DOOR_ROI": np.array([(192, 599), (977, 207), (1170, 298), (1521, 1608), (893, 2068)], np.int32),
            "STANDING_ZONE": np.array([(278, 996), (1020, 636), (1372, 1563), (720, 1914)], np.int32),
            "LOCKS_ROI": np.array([(1346, 1518), (1640, 1728), (1169, 2092), (918, 1811)], np.int32),
            "DOOR_CORNER_ROI": np.array([(409.72072155070737, 479.8188706883842), (487.6075140035375, 673.6301914431011), (527.456570607311, 497.932078235554), (715.833929097877, 374.76226691479934), (695.9094007959902, 325.8566065374409)], np.int32),
            "INTERACTION_ZONE": np.array([(42, 709), (114, 2094), (3316, 2099), (3628, 505), (2197, 25)], np.int32)
        }
    }
    ,
    #Andhra Pradesh streams Below

    
    # Stream 7
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.76.17:8001/Streaming/Channels/2001",
        "camera_id": "GF-4-CAM-20",
        "site_id": "4",
        "site_name": "vijayawada",
        "closed_door_reference": "close_doors/closed_GF-4-CAM-20.jpg",
        "ssim_threshold": 0.653,
        "debounce_threshold": 5,
        "intensity_threshold": 10,
        "motion_threshold": 3.0,
        # Full-door corner ROI (DOOR_CORNER_ROI == DOOR_ROI): camera shows only half the
        # door so the whole door is the open/close patch. A 0.3 min-visible let a 30%
        # unoccluded sliver drive the transition -> flip-prone. Raised to 0.45 so a real
        # door-surface fraction must be visible to trust the reading; corruption false-opens
        # are handled separately by the frame-quality cell-block gate.
        "door_corner_min_visible_ratio": 0.45,
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
    # Stream 8
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@106.51.52.103:8002/Streaming/Channels/2101",
        "camera_id": "FF-3-CAM-21",
        "site_id": "3",
        "site_name": "vizag",
        "closed_door_reference": "close_doors/closed_FF-3-CAM-21.jpg",
        "ssim_threshold": 0.714,
        "debounce_threshold": 7,
        "intensity_threshold": 12,
        "motion_threshold": 5.0,
        # Full-door corner ROI (see vijayawada note): raised 0.3 -> 0.45 to stop a partial
        # sliver flipping the transition. Corruption false-opens handled by the cell-block gate.
        "door_corner_min_visible_ratio": 0.45,
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
    # Stream 9 - Guntur
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@49.205.164.154:8001/Streaming/Channels/1901",
        "camera_id": "GF-19-CAM-19",
        "site_id": "19",
        "site_name": "guntur-store",
        "closed_door_reference": "close_doors/closed_GF-19-CAM-19.jpg",
        "ssim_threshold": 0.694,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 1.5,
        "max_unlock_seconds": 10.0,
        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "DOOR_CORNER_ROI": np.array([[1914, 4], [1901, 52], [2084, 4]], np.int32),
            "INTERACTION_ZONE": np.array([[29, 14], [2639, 19], [2649, 1477], [39, 1487]], np.int32),
            "LOCKS_ROI": np.array([[2294, 461], [1833, 140], [1667, 524], [2056, 835]], np.int32),
            "DOOR_ROI": np.array([[1867, 19], [2494, 189], [1925, 1054], [1453, 646]], np.int32),
            "STANDING_ZONE": np.array([[1638, 646], [1419, 811], [1764, 1156], [1964, 947]], np.int32)
        }
    },
    # Stream 10 - Kakinada
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.145.50:8001/Streaming/Channels/1501",
        "camera_id": "GF-22-CAM-15",
        "site_id": "22",
        "site_name": "kakinada-store",
        "closed_door_reference": "close_doors/closed_GF-22-CAM-15.jpg",
        "ssim_threshold": 0.872,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 5.0,
        "max_unlock_seconds": 10.0,
        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "LOCKS_ROI": np.array([[2366, 483], [3161, 770], [2893, 1643], [2108, 1316]], np.int32),
            "DOOR_CORNER_ROI": np.array([[2593, 213], [2741, 249], [2591, 268]], np.int32),
            "STANDING_ZONE": np.array([[2437, 1370], [2924, 1560], [2817, 1833], [2252, 1550]], np.int32),
            "DOOR_ROI": np.array([[2326, 62], [3261, 332], [3022, 1689], [2286, 1396]], np.int32),
            "INTERACTION_ZONE": np.array([[45, 39], [3791, 44], [3786, 2126], [45, 2111]], np.int32)
        }
    },
    # Stream 11 - Rajahmundry
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@106.51.52.101:8001/Streaming/Channels/1501",
        "camera_id": "GF-14-CAM-15",
        "site_id": "14",
        "site_name": "rajahmundry-store",
        "closed_door_reference": "close_doors/closed_GF-14-CAM-15.jpg",
        "ssim_threshold": 0.832,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 5.0,
        "max_unlock_seconds": 10.0,
        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "DOOR_CORNER_ROI": np.array([[1433.0049806308791, 4.796163069544375], [1428.8729016786563, 132.89061058845226], [1511.514480723113, 49.422615753550986], [1627.2126913853522, 48.59619996310642], [1625.559859804463, 3.143331488655241]], np.int32),
            "INTERACTION_ZONE": np.array([[32, 34], [2656, 31], [2653, 1489], [38, 1496]], np.int32),
            "STANDING_ZONE": np.array([[1980, 968], [1568, 992], [1462, 669], [1888, 676]], np.int32),
            "LOCKS_ROI": np.array([[1441, 124], [1901, 124], [1869, 555], [1438, 551]], np.int32),
            "DOOR_ROI": np.array([[1390, 13], [1904, 5], [1895, 789], [1423, 795]], np.int32)
        }
    },
    # Stream 12 - Kurnool
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@103.206.112.148:8001/Streaming/Channels/1801",
        "camera_id": "GF-16-CAM-18",
        "site_id": "16",
        "site_name": "kurnool-store",
        "closed_door_reference": "close_doors/closed_GF-16-CAM-18.jpg",
        "ssim_threshold": 0.742,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 5.0,
        "max_unlock_seconds": 10.0,
        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "INTERACTION_ZONE": np.array([[30, 34], [2651, 31], [2658, 1482], [40, 1482]], np.int32),
            "DOOR_CORNER_ROI": np.array([[1946.656660412758, 23.6998123827392], [2002.1313320825516, 810.4315196998124], [1043.9324577861164, 603.6622889305817], [791.7748592870544, 0]], np.int32),
            "STANDING_ZONE": np.array([[1279, 642], [1066, 1077], [1787, 1318], [1968, 824]], np.int32),
            "LOCKS_ROI": np.array([[858.1294964028773, 8.870504365550557], [1885.1095178544658, 10.03356494491909], [1970.0719424460422, 788.4388496892911], [1059.06191369606, 558.2739212007505]], np.int32),
            "DOOR_ROI": np.array([[1946.656660412758, 23.6998123827392], [2002.1313320825516, 810.4315196998124], [1043.9324577861164, 603.6622889305817], [791.7748592870544, 0]], np.int32)
        }
    },
    # Stream 13 - Ongole
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.81.163:8001/Streaming/Channels/501",
        "camera_id": "GF-27-CAM-5",
        "site_id": "27",
        "site_name": "ongole-store",
        "closed_door_reference": "close_doors/closed_GF-27-CAM-05.jpg",
        "ssim_threshold": 0.581,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 5.0,
        "max_unlock_seconds": 10.0,
        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "STANDING_ZONE": np.array([[1955, 957], [2341, 1200], [2221, 1438], [1811, 1156]], np.int32),
            "DOOR_CORNER_ROI": np.array([[2217, 277], [2330, 324], [2207, 326]], np.int32),
            "INTERACTION_ZONE": np.array([[60, 48], [47, 1479], [2638, 1486], [2635, 41]], np.int32),
            "LOCKS_ROI": np.array([[2123, 470], [2593, 676], [2421, 1143], [1975, 916]], np.int32),
            "DOOR_ROI": np.array([[2157, 216], [2675, 446], [2294, 1438], [1828, 1153]], np.int32)
        }
    },
    # Stream 14 - Srikakulam
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.130.104:8001/Streaming/Channels/2701",
        "camera_id": "GF-20-CAM-27",
        "site_id": "20",
        "site_name": "srikakulam-store",
        "closed_door_reference": "close_doors/closed_GF-20-CAM-27.jpg",
        "ssim_threshold": 0.772,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 5.0,
        "max_unlock_seconds": 10.0,
        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "INTERACTION_ZONE": np.array([[45, 64], [3790, 49], [3800, 2111], [40, 2106]], np.int32),
            "DOOR_CORNER_ROI": np.array([[545.7116015747183, 541.2324114892866], [807.9574570642396, 421.1198059215669], [829.9781014183216, 475.1704784270408], [656.882490659258, 553.7512846865362], [593.2990716801556, 711.6135662898249]], np.int32),
            "STANDING_ZONE": np.array([[942, 1775], [1142, 2082], [1786, 1638], [1562, 1399]], np.int32),
            "LOCKS_ROI": np.array([[421, 911], [1288, 495], [1571, 1224], [793, 1706]], np.int32),
            "DOOR_ROI": np.array([[301, 461], [1246, 30], [1648, 1474], [920, 1902]], np.int32)
        }
    },
    # Stream 15 - Tirupathi
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@202.83.31.24:8001/Streaming/Channels/2101",
        "camera_id": "GF-23-CAM-21",
        "site_id": "23",
        "site_name": "tirupathi-store",
        "closed_door_reference": "close_doors/closed_GF-23-CAM-21.jpg",
        "ssim_threshold": 0.646,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 5.0,
        "max_unlock_seconds": 10.0,
        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "DOOR_CORNER_ROI": np.array([[2016.9267951023783, 10.108836008116578], [2011.3788311298063, 127.14078374455725], [2067.982604714712, 53.991291727140755], [2203.8316613184857, 51.37880986937588], [2204.702488604407, 7.837445573294626]], np.int32),
            "INTERACTION_ZONE": np.array([[52, 68], [3766, 46], [3781, 2084], [59, 2114]], np.int32),
            "LOCKS_ROI": np.array([[2332.853717026378, 264.33710174717356], [1487.0101169064742, 58.12949640287767], [1580.938174460431, 612.4892086330933], [2280.7942895683445, 820.6043165467622]], np.int32),
            "STANDING_ZONE": np.array([[1996, 1375], [2144, 1056], [1778, 785], [1523, 953]], np.int32),
            "DOOR_ROI": np.array([[1591, 24], [2390, 26], [2230, 1153], [1591, 874]], np.int32)
        }
    },
    # Stream 16 - Bhimavaram
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.94.133:8001/Streaming/Channels/3001",
        "camera_id": "GF-25-CAM-30",
        "site_id": "25",
        "site_name": "bhimavaram-store",
        "closed_door_reference": "close_doors/closed_GF-25-CAM-30.jpg",
        "ssim_threshold": 0.644,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 5.0,
        "max_unlock_seconds": 10.0,
        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "INTERACTION_ZONE": np.array([[55.964125560538115, 64.03587443946188], [3792.6457399103138, 55.426008968609864], [3788.3408071748877, 2113.183856502242], [55.964125560538115, 2100.269058295964]], np.int32),
            "DOOR_CORNER_ROI": np.array([[1292.3076923076915, 0], [1494.3635601716098, 5.381165919282507], [1293.9708939708933, 74.01247401247397]], np.int32),
            "LOCKS_ROI": np.array([[1203.7729777115803, 168.21741825289763], [1932.7897642823236, 104.26857732563946], [2022.3181415804852, 728.4092647756793], [1288.1854477355612, 879.3285293640085]], np.int32),
            "STANDING_ZONE": np.array([[1371.2570295399462, 1125.1281929135391], [2047.679793767588, 752.8955475101623], [2303.839893830127, 921.0006131762034], [1619.4121264755308, 1373.2832898491238]], np.int32),
            "DOOR_ROI": np.array([[1172.5119142773647, 1.5401540154015394], [1303.938390258296, 1144.481114778144], [2071.375133932663, 1031.8298496516313], [2050.253021721442, 3.887055372203885]], np.int32)
        }
    },
    # Stream 17 - Vizianagaram
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@106.51.52.2:8001/Streaming/Channels/1301",
        "camera_id": "GF-29-CAM-13",
        "site_id": "29",
        "site_name": "vizianagaram-store",
        "closed_door_reference": "close_doors/closed_GF-29-CAM-13.jpg",
        "ssim_threshold": 0.713,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 5.0,
        "max_unlock_seconds": 10.0,
        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "DOOR_CORNER_ROI": np.array([[2301.022594424459, 433.1779310926256], [2477.2144409472407, 627.6335665842322], [2240.8595248800943, 547.058027015887]], np.int32),
            "INTERACTION_ZONE": np.array([[29.007194244604317, 41.26618705035976], [2655.7697841726617, 28.37410071942451], [2655.7697841726617, 1478.7338129496404], [32.23021582733813, 1501.294964028777]], np.int32),
            "LOCKS_ROI": np.array([[2530.071942446043, 1488.4028776978419], [1717.8705035971223, 563.3956834532374], [1434.2446043165467, 1146.7625899280577], [1543.8273381294964, 1514.1870503597122]], np.int32),
            "DOOR_ROI": np.array([[2272.2302158273383, 266.8776978417267], [2671.884892086331, 721.3237410071943], [2658.9928057553957, 1488.4028776978419], [1166.7338129496402, 1514.1870503597122], [1685.6402877697842, 286.21582733812954]], np.int32),
            "STANDING_ZONE": np.array([[1457.4709193245778, 1334.9193245778613], [1110.9811320754718, 1345.5345911949685], [1127.7106918238994, 1520], [1507.7106918238994, 1520]], np.int32)
        }
    },
    # Karnataka streams Below

    # Stream 18 - Jayanagar
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@106.51.37.109:8001/Streaming/Channels/2501",
        "camera_id": "GF-5-CAM-25",
        "site_id": "5",
        "site_name": "jayanagar",
        "closed_door_reference": "close_doors/closed_GF-5-CAM-25.jpg",
        "ssim_threshold": 0.796,
        "debounce_threshold": 15,
        "intensity_threshold": 8,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 4.0,
        "max_unlock_seconds": 10.0,

        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "DOOR_CORNER_ROI": np.array([(806.9950193691197, 2.235749861649161), (794.1855746172288, 97.02416717467368), (1075.9933591588263, 2.235749861649161)], np.int32),
            "DOOR_ROI": np.array([(581, 3), (1187, 0), (1034, 742), (471, 797)], np.int32),
            "LOCKS_ROI": np.array([(526, 81), (1144, 49), (1106, 557), (501, 588)], np.int32),
            "STANDING_ZONE": np.array([(545, 897), (544, 1103), (1165, 1050), (1154, 853)], np.int32),
            "INTERACTION_ZONE": np.array([(0, 8), (0, 1340), (1902, 1416), (2076, 11)], np.int32)
        }
    },
    # Stream 19 - UBCity Bangalore
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@106.51.89.106:8002/Streaming/Channels/401",
        "camera_id": "FF-34-CAM-4",
        "site_id": "34",
        "site_name": "ub-city-banglore-store",
        "closed_door_reference": "close_doors/closed_FF-34-CAM-04.jpg",
        "ssim_threshold": 0.863,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 5.0,
        "max_unlock_seconds": 10.0,

        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "DOOR_CORNER_ROI": np.array([(1606.788415421508, 4.205866076369671), (1772.0715735104213, 0.664084117321527), (1592.6212875853153, 64.41615938018812)], np.int32),
            "INTERACTION_ZONE": np.array([(50.431519699812384, 64.16510318949344), (3775.1594746716696, 56.96060037523452), (3789.5684803001873, 2124.6529080675423), (50.431519699812384, 2095.8348968105065)], np.int32),
            "LOCKS_ROI": np.array([(1558.1007264500893, 111.79856115107908), (2223.4245025854307, 148.6330935251798), (2096.8057975494594, 595.2517985611508), (1500.546769615557, 595.2517985611508)], np.int32),
            "DOOR_ROI": np.array([(1559.1890124264219, 9.548724656638322), (2074.035317200784, 3.270111183780247), (1958.9274035317192, 750.4251144538911), (1446.1739699149764, 794.3754087638977)], np.int32),
            "STANDING_ZONE": np.array([(1894.885533334695, 721.1249182472201), (2041.38651436805, 886.4617396991493), (1589.3263443222688, 978.5480706344011), (1468.4086445185906, 785.6387514924097)], np.int32)
        }
    },
    # Stream 20 - Marthahalli
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@106.51.66.143:8001/Streaming/Channels/1701",
        "camera_id": "GF-37-CAM-17",
        "site_id": "37",
        "site_name": "marathahalli-banglore-store",
        "closed_door_reference": "close_doors/closed_GF-37-CAM-17.jpg",
        "ssim_threshold": 0.721,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 5.0,
        "max_unlock_seconds": 10.0,

        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "STANDING_ZONE": np.array([(2699.773975462486, 596.1973278520038), (2535.3340987923934, 770.5035971223017), (2956.30018306783, 967.8314491264126), (3092.7852807040067, 737.6156217882833)], np.int32),
            "INTERACTION_ZONE": np.array([(41.43884892086331, 0), (3780.143884892086, 0), (3783.8273346166816, 2115.971223021583), (49.72661519222122, 2115.971223021583)], np.int32),
            "DOOR_CORNER_ROI": np.array([(2612.4184306401016, 3.0252720900202896), (2777.701588729015, 0.664084117321527), (2646.655656244234, 80.94447518907945)], np.int32),
            "LOCKS_ROI": np.array([(3018.7574088456495, 32.57030739045126), (3146.42254946043, 566.2524525833876), (2443.5379086436724, 554.0414727041895), (2227.652111265532, 36.756049705689975)], np.int32),
            "DOOR_ROI": np.array([(2612.4184306401016, 3.0252720900202896), (2777.701588729015, 0.664084117321527), (2646.655656244234, 80.94447518907945)], np.int32)
        }
    },
    # Tamil Nadu streams Below

    # Stream 21 - Coimbatore
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.251.16:8001/Streaming/Channels/2701",
        "camera_id": "GF-40-CAM-27",
        "site_id": "40",
        "site_name": "coimbatore-store",
        "closed_door_reference": "close_doors/closed_GF-40-CAM-27.jpg",
        # PROVISIONAL 2026-06-10: measured evening closed-door SSIM = 0.93 vs threshold
        # 0.80 → healthy +0.13 margin, door state correct (no false-open). Threshold
        # kept; added open_hysteresis as an extra bias-toward-CLOSED guard. Confirm from
        # VIDEO_FRAME logs.
        "ssim_threshold": 0.80,
        "door_open_hysteresis": 0.06,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 5.0,
        "max_unlock_seconds": 10.0,
        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "DOOR_CORNER_ROI": np.array([[558, 1], [543, 62], [769, 1]], np.int32),
            "STANDING_ZONE": np.array([[442, 550], [417, 743], [876, 727], [847, 541]], np.int32),
            "LOCKS_ROI": np.array([[507, 106], [440, 527], [847, 527], [973, 131]], np.int32),
            "DOOR_ROI": np.array([[525, 3], [427, 662], [835, 653], [1070, 10]], np.int32),
            "INTERACTION_ZONE": np.array([[11, 19], [1899, 22], [1899, 1055], [20, 1060]], np.int32)
        }
    },
    # Stream 22 - Chennai Annanagar
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@49.207.187.203:8001/Streaming/Channels/1501",
        "camera_id": "GF-41-CAM-15",
        "site_id": "41",
        "site_name": "chennai-annanagar-store",
        "closed_door_reference": "close_doors/closed_GF-41-CAM-15.jpg",
        "ssim_threshold": 0.80,
        "debounce_threshold": 15,
        "intensity_threshold": 6,
        "motion_threshold": 3.0,
        "door_corner_min_visible_ratio": 0.5,
        "min_unlock_seconds": 5.0,
        "max_unlock_seconds": 10.0,
        "evening_second_unlocker_timeout_seconds": 300.0,
        "rois": {
            "STANDING_ZONE": np.array([[2421, 1310], [2071, 1614], [2785, 2129], [3066, 1779]], np.int32),
            "INTERACTION_ZONE": np.array([[50, 57], [3798, 34], [3798, 2102], [50, 2111]], np.int32),
            "LOCKS_ROI": np.array([[3355.9943794964015, 315.9712230215826], [3017.116681654675, 131.79856115107907], [2947.131070143884, 328.86330935251783]], np.int32),
            "DOOR_ROI": np.array([[3651, 389], [3061, 1867], [2223, 1402], [2527, 7]], np.int32),
            "DOOR_CORNER_ROI": np.array([[2899, 111], [2879, 203], [3192, 182]], np.int32)
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
