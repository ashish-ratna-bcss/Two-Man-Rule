# Two-Man Rule Dual-Control Security System

High-security monitoring system using YOLOv11-Pose and ByteTrack to enforce dual-control (two-man rule) access to restricted resources.

## Overview

This system ensures a restricted locker/door can only be opened when **exactly two authorized individuals** have each performed a **10-second mechanical unlocking sequence** on their respective locks. It uses pose detection, ID tracking, and state machine logic to enforce strict security rules.

### Key Features

- **Dual-Auth Requirement**: Exactly 2 different people must each activate a lock
- **10-Second Dwell Timer**: Each person must maintain hand contact with their lock for 10 seconds
- **Occlusion Handling**: Kinematic fallback uses elbow proximity when hands are obscured
- **Overcrowd Detection**: Immediately resets timers if >2 people enter the interaction zone
- **Evidence Capture**: Automatic screenshot capture on success or violation
- **SSIM-Based Verification**: Door state detection via baseline image comparison

## System Architecture

Video Input → Pose Detection → Tracking → Occupancy Census → State Machine → Door Verification → Visualization & Alerts

### Components

| Module | Responsibility |
|--------|-----------------|
| config.py | ROI definitions, constants, model paths |
| models/pose_detector.py | YOLOv11-Pose skeleton detection |
| models/tracker.py | ByteTrack persistent ID assignment |
| models/door_verifier.py | SSIM-based door state verification |
| logic/roi_manager.py | ROI intersection and distance calculations |
| logic/state_machine.py | Dual-auth state logic and timer management |
| logic/kinematic_fallback.py | Occlusion handling via shoulder/elbow |
| io/visualizer.py | Overlay rendering (progress bars, bboxes, text) |
| io/alert_system.py | Screenshot capture and event logging |
| io/video_handler.py | Video I/O wrapper |
| main.py | Pipeline orchestration |

## Setup

See SETUP.md for detailed installation instructions.

## Usage

```bash
python main.py <video_file_or_webcam>
```

Exit with `q` key.

## Configuration

Before running, configure ROIs in config.py:

- LOCK_A_ROI: Rectangle (x, y, w, h) for Lock A
- LOCK_B_ROI: Rectangle (x, y, w, h) for Lock B
- DOOR_ROI: Polygon points for door/locker region
- INTERACTION_ZONE: Large floor-level polygon
- CLOSED_DOOR_REFERENCE: Path to baseline closed-door image

## Output

- logs/evidence/SUCCESS_*.jpg: Successful dual-auth verification
- logs/evidence/VIOLATION_SOLO_*.jpg: Violation (only 1 person authorized)
- logs/evidence/VIOLATION_OVERCROWD_*.jpg: Violation (>2 people detected)
- logs/session_*.json: Detailed event log
