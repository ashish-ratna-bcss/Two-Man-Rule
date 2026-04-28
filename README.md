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
| `config.py` | Camera metadata, ROI polygons, and security thresholds. |
| `main.py` | Master Orchestrator, IST Scheduling, and Capture logic. |
| `models/door_verifier.py` | SSIM-based patch comparison for door corner state. |
| `logic/state_machine.py` | Dual-auth sequence management and lock interaction timers. |
| `models/pose_detector.py` | Skeleton tracking using YOLO Pose models. |
| `models/tracker.py` | ID persistence with ByteTrack and Re-ID recovery. |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Production Run (Headless)
The script is pre-configured with the vault's RTSP stream and will automatically manage the daily schedule:
```bash
python3 main.py
```

### 2. Manual Preview (GUI enabled)
Useful for calibrating ROIs or verifying live performance:
```bash
python3 main.py --show
```

### 3. File Processing (Testing)
Run against a recorded clip for offline validation:
```bash
python3 main.py "path/to/video.mp4" --show
```

## Evidence Output Structure

Forensic screenshots are stored in the root `StrongRoomCheck` directory:
```text
StrongRoomCheck/
├── ROI_PREVIEW_GF-1-CAM-40.jpg (Current ROI configuration)
└── 2026-04-28/
    ├── MorningCheck/
    │   └── StrongRoomCheck_Morning_GF-1-CAM-40_20260428_093512_456.png
    └── EveningCheck/
        └── StrongRoomCheck_Evening_GF-1-CAM-40_224510_123.png
```

## Configuration

Update `config.py` to adjust:
- **RTSP_URLS**: Site metadata and camera stream links.
- **Time Windows**: Adjust the morning/evening schedule.
- **ROIs**: Redefine lock coordinates and standing zones.
- **SSIM_THRESHOLD**: Fine-tune door state sensitivity (default is 0.92).

---
*Developed for PMJ - Two-Man Rule Security Compliance*
