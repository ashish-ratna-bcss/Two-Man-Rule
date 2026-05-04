# Two-Man Rule Dual-Control Security System

High-security monitoring system using YOLO-Pose and ByteTrack to enforce a robust two-man rule for restricted door/vault access. This system operates as a persistent daily daemon, monitoring specific time windows and capturing forensic evidence of authorized and unauthorized access.

## Key Features

- **Daily Orchestration (IST)**: Operates on a strict schedule based on Indian Standard Time:
  - **Morning Open Check (09:30 AM – 10:30 AM)**: Monitors vault opening.
  - **Evening Close Check (08:30 PM – 11:00 PM)**: Monitors vault closure.
- **Biomechanical Dual-Auth**: Requires two different individuals to perform specific lock interactions simultaneously (6-10 second dwell time, arms raised, correct positioning).
- **SSIM Door Verification**: Uses Grayscale Structural Similarity (SSIM) to detect door state (OPEN vs. CLOSED) with 10-frame temporal debouncing to prevent false triggers.
- **Dynamic 5s Grace Period**: Upon a door transition, the system provides a 5-second window for unlockers to be properly positioned in the Interaction Zone. If auth is met within this window, a capture is taken immediately.
- **Hierarchical Evidence Storage**: Organizes captures in a `StrongRoomCheck/YYYY-MM-DD/` folder structure with subfolders for `MorningCheck` and `EveningCheck`.
- **Headless Operation**: Optimized for background server deployment with no window display by default.

## System Architecture

The pipeline integrates pose detection, multi-sensor tracking, and state-machine logic to ensure zero-trust security.

| Module | Responsibility |
|--------|-----------------|
| `config.py` | Stream metadata (`STREAMS_CONFIG`), ROI polygons, reference paths, and thresholds. |
| `main.py` | Master Orchestrator, Stream Selection, IST Scheduling, and Capture logic. |
| `models/door_verifier.py` | SSIM-based patch comparison for door corner state. |
| `logic/state_machine.py` | Dual-auth sequence management and lock interaction timers. |
| `models/pose_detector.py` | Skeleton tracking using YOLO Pose models. |
| `models/tracker.py` | ID persistence with ByteTrack and Re-ID recovery. |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Production Run (Headless, Daemon)
Reads live RTSP stream for a specific configuration, runs forever, resets daily at midnight IST:
```bash
# Run the first stream (default)
python3 main.py

# Run the second stream defined in config.py
python3 main.py --stream-index 1
```
**Behaviour:**
- **Morning window 09:30–10:30 IST**: Tracks 2 unlockers, waits for door CLOSED→OPEN transition
  - Screenshot captured at transition moment
  - `2 Persons: Available` — both verified unlockers in interaction zone when door opened
  - `2 Persons: Unavailable` — door opened without proper dual auth
- **Evening window 20:30–23:00 IST**: Waits for door OPEN→CLOSED, then tracks for 2 unlockers (5-min timeout)
  - Screenshot captured when auth confirmed or timeout reached
  - `2 Persons: Available` — both unlockers verified within timeout
  - `2 Persons: Unavailable` — timeout elapsed without dual auth
- **Midnight IST**: All flags reset, cycle repeats next day
- **RTSP drops**: Auto-reconnects every 5s, script never dies

### 2. Test Window — Morning
Forces morning window logic regardless of current IST time. Exits automatically after check criteria met and screenshot saved:
```bash
# Test using Stream 0's ROIs
python3 main.py 28-E.mp4 --test-window morning --show

# Test using Stream 1's ROIs
python3 main.py 28-E.mp4 --stream-index 1 --test-window morning --show
```

### 3. Test Window — Evening
Forces evening window logic. Exits automatically after check complete:
```bash
python3 main.py 28-E.mp4 --stream-index 0 --test-window evening --show
```

### 4. Test Window — Debug Mode
All overlays visible on screen (ROI polygons, SSIM, AI ms, progress bars, pose keypoints, all detections). Screenshots remain clean regardless:
```bash
python3 main.py 28-E.mp4 --stream-index 0 --test-window morning --show --debug
```

### 5. Performance Tuning
```bash
# Run inference every frame (full accuracy, slower)
python3 main.py --process-every 1

# Force GPU
python3 main.py --device cuda

# Disable half-precision (if GPU precision issues)
python3 main.py --device cuda --no-half
```

### 6. ROI Calibration on Downscaled Clips
If test video resolution differs from RTSP-calibrated 2688×1520:
```bash
python3 main.py "path/to/video.mp4" --scale-rois --show --debug
```

### Screenshot Content (always clean, client-facing)
Every captured screenshot contains only:
- Raw camera frame (no debug overlays)
- Bounding boxes on P1/P2 verified unlockers (green = authorized, yellow = in progress)
- Bottom-left panel: `2 Persons: Available` (green) or `2 Persons: Unavailable` (red)
- Bottom-right panel: `Door: Open` (red) or `Door: Closed` (green)

### All Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `video_source` | config | (Positional) Override video path/URL for the stream. |
| `--stream-index N` | 0 | Index of the stream configuration to use from `config.STREAMS_CONFIG`. |
| `--show` | off | Enable live OpenCV preview window |
| `--debug` | off | Show all debug overlays on live window. Screenshots always clean |
| `--test-window morning\|evening` | off | Force auth window regardless of IST time. Exits after check complete |
| `--process-every N` | 3 | Run pose inference every N frames |
| `--device auto\|cuda\|cpu` | auto | Inference device |
| `--no-half` | off | Disable CUDA half-precision |
| `--scale-rois` | off | Scale RTSP-calibrated ROIs to a different video resolution |
| `--show-all-detections` | off | Show all raw detections on live window (subset of `--debug`) |

## Evidence Output Structure

Forensic screenshots are stored dynamically based on the stream's store and camera configuration:
```text
strong_room_opening/
└── somajiguda/
    └── GF-1-CAM-40/
        ├── ROI_PREVIEW_GF-1-CAM-40.jpg
        └── 28-04-2026/
            ├── alert_1_GF-1-CAM-40_28-04-2026_09-35-12.png
            └── alert_2_GF-1-CAM-40_28-04-2026_22-45-10.png
```

## Configuration

Update `config.py` to adjust:
- **STREAMS_CONFIG**: List of isolated dictionaries for each store/floor. Defines `rtsp_url`, specific `rois`, and the `closed_door_reference` image path.
- **Time Windows**: Adjust the morning/evening schedule.
- **SSIM_THRESHOLD**: Fine-tune door state sensitivity (default is 0.92).

---
*Developed for PMJ - Two-Man Rule Security Compliance*
