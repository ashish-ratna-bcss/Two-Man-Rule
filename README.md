# Two-Man Rule Dual-Control Security System

High-security monitoring system using YOLOv11-Pose and ByteTrack to enforce dual-control (two-man rule) access to restricted resources.

## Overview

This system ensures a restricted locker/door can only be opened after **two different individuals** sequentially perform a valid lock interaction at the door. Each unlocker must stand in the standing zone, keep their head inside the interaction zone, face the door with both arms raised toward the locks, and interact with both lock ROIs for **6-10 seconds**.

### Key Features

- **Sequential Dual-Auth Requirement**: 2 different unlockers must complete the same lock interaction flow
- **6-10 Second Unlock Timer**: Each unlocker must hold the qualifying pose long enough to count
- **Head-Based Assignment**: Only people with head keypoints inside the interaction zone can become assigned unlockers
- **Hand/Elbow Validation**: Right and left wrists/elbows are checked against both lock ROIs for robust back-facing operation
- **Ignored Bystanders**: People already present or just entering are not assigned unless they perform the unlock pose
- **Evidence Capture**: Automatic screenshot capture on success or violation
- **SSIM-Based Verification**: Door state detection via baseline image comparison

## System Architecture

Video Input → Pose Detection → Tracking → Occupancy Census → State Machine → Door Verification → Visualization & Alerts

### Components

| Module | Responsibility |
|--------|-----------------|
| config.py | ROI definitions, constants, model paths |
| models/pose_detector.py | YOLOv11-Pose skeleton detection |
| models/tracker.py | ByteTrack tracking used internally for unlocker identity |
| models/door_verifier.py | SSIM-based door state verification |
| logic/roi_manager.py | ROI intersection and distance calculations |
| logic/state_machine.py | Dual-auth state logic and timer management |
| logic/kinematic_fallback.py | Occlusion handling via shoulder/elbow |
| io_/visualizer.py | Overlay rendering (progress bars, bboxes, text) |
| io_/alert_system.py | Screenshot capture and event logging |
| io_/video_handler.py | Video I/O wrapper |
| main.py | Pipeline orchestration |

## Setup

See SETUP.md for detailed installation instructions.

## Usage

```bash
python main.py Strong-Room.mp4
```

The run opens a live ROI/debug window by default. Press `q` to stop. The configured polygons are used as raw RTSP-captured coordinates for the `2688x1520` strong-room stream. For smoother preview, pose inference runs every 3 frames by default while playback still displays every frame and timers are compensated. GPU is used automatically when CUDA is available; use `--device cuda` to force it. The UI shows tracking IDs only for active/verified unlockers, not every person in the room; use `--show-all-detections` only when calibrating raw detections. Use `--process-every 1` for full per-frame analysis, `--scale-rois` only for downscaled clips such as `test_video.mp4`, and `--no-show` for headless runs.

## Configuration

Before running, configure ROIs in config.py:

- LOCK_A_ROI: Rectangle (x, y, w, h) for Lock A
- LOCK_B_ROI: Rectangle (x, y, w, h) for Lock B
- DOOR_ROI: Polygon points for door/locker region
- INTERACTION_ZONE: Large floor-level polygon
- CLOSED_DOOR_REFERENCE: Path to baseline closed-door image

## Output

- logs/evidence/DUAL_AUTH_SUCCESS_*.jpg: Successful two-person unlock sequence
- logs/evidence/CRITICAL_VIOLATION_LONE_WOLF_*.jpg: Door opened before two valid unlockers completed
- logs/evidence/VIOLATION_OVERCROWD_*.jpg: Violation (>2 people detected)
- logs/session_*.json: Detailed event log
