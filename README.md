# Two-Man Rule Dual-Control Security System

High-security monitoring system using YOLO-Pose and ByteTrack to enforce a robust two-man rule for restricted door/vault access. This system operates as a persistent daily daemon, monitoring specific time windows and capturing forensic evidence of authorized and unauthorized access.

## Key Features

- **Daily Orchestration (IST)**: Operates on a strict schedule based on Indian Standard Time:
  - **Morning Open Check (07:00 AM – 11:00 AM)**: Monitors vault opening.
  - **Evening Close Check (08:00 PM – 11:00 PM)**: Monitors vault closure.
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
# Run all configured streams in parallel (default)
python3 main.py

# Run one stream by index
python3 main.py --stream-index 1

# Run selected streams only
python3 main.py --stream-indices 0,3,4
```
**Behaviour:**
- **Morning window 07:00–11:00 IST**: Tracks 2 unlockers, waits for door CLOSED→OPEN transition
  - Screenshot captured at transition moment
  - `2 Persons: Available` — both verified unlockers in interaction zone when door opened
  - `2 Persons: Unavailable` — door opened without proper dual auth
- **Evening window 20:00–23:00 IST**: Waits for door OPEN→CLOSED, then tracks for 2 unlockers (5-min timeout)
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
| `--stream-index N` | none | Single stream index from `config.STREAMS_CONFIG`. |
| `--stream-indices a,b,c` | none | Run only selected stream indexes in parallel (example: `0,2,4`). |
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

## Parameter Tuning Guide

Here's a practical decision tree for adjusting each parameter based on real-world conditions you'll observe:

### **1. SSIM Threshold** – Door Open/Close Detection

#### **Problem: Door opens but NOT detected (False Negative)**
**Increase the threshold slightly** (e.g., 0.65 → 0.70)
- **Conditions:** Door physically opens but SSIM stays high (>0.75)
- **Why:** Low-texture door surface, consistent lighting, minimal visual change
- **Example:** Dark flat door, camera angle shows little edge detail when opening
- **Check:** Run debug mode, log SSIM values when door actually opens

#### **Problem: Door appears to open when it's really closed (False Positive)**
**Decrease the threshold slightly** (e.g., 0.65 → 0.60)
- **Conditions:** Shadows, reflections, or lighting artifacts cause SSIM to dip below threshold
- **Why:** Temporary shadow/glare makes patch look different without actual door movement
- **Example:** Sun reflection on door, overhead light flickers, Person's shadow crosses DOOR_CORNER_ROI
- **Check:** Look for brief SSIM dips during non-opening events

#### **Problem: Oscillating door state (flickering between open/closed)**
**Increase threshold + increase debounce** (see below)
- **Conditions:** SSIM hovers right around your threshold (e.g., 0.64-0.66)
- **Why:** Micro-variations in lighting cause SSIM to cross threshold repeatedly
- **Solution:** Move threshold further from natural variation range

### **2. Debounce Threshold** – State Change Confirmation

#### **Problem: Door opens but detection is delayed/slow**
**Decrease debounce frames** (e.g., 20 → 10–15)
- **Conditions:** Real door opens but system takes 0.67s to confirm
- **Why:** 20 frames at 30 FPS = 0.67s lag; if your operations are faster, it's too slow
- **Example:** Security team needs instant alerts; 0.67s feels sluggish
- **Risk:** More false positives from noise

#### **Problem: False door-open alerts from shadows/flickers**
**Increase debounce frames** (e.g., 20 → 25–30)
- **Conditions:** Doorway has lots of transient shadows, reflections, or light changes
- **Why:** Requires 20+ consecutive SSIM dips to confirm; brief artifacts don't count
- **Example:** Windows near door, passing traffic causing shadows, flickering fluorescent lights
- **Trade-off:** Real door openings take 0.83–1.0s to confirm

#### **Problem: Door closes but system still thinks it's open**
**Decrease debounce slightly** (e.g., 20 → 15)
- **Conditions:** When door reaches "closed" SSIM level, it takes too long to register
- **Why:** 20 frames required SSIM to stay high again
- **Check:** Log state transitions; if close detection lags, debounce is too high

### **3. Intensity Threshold** – Brightness Change Detection

#### **Problem: System triggers "door open" when lights just turn on/off**
**Increase intensity threshold** (e.g., 35 → 45–50)
- **Conditions:** Overhead lights or sudden illumination in room
- **Why:** Lights on/off causes +40 brightness units in DOOR_CORNER_ROI, confuses door detector
- **Example:** Night shift → morning shift, lights flip on; system falsely detects door movement
- **Mitigation:** `DOOR_DARKENING_PROTECTION = True` should help, but increase threshold as backup

#### **Problem: Genuine door movement not detected (especially subtle openings)**
**Decrease intensity threshold** (e.g., 35 → 25–30)
- **Conditions:** Door opening is slow/partial, causes only ±15–20 brightness units change
- **Why:** Threshold too high; real motion is ignored
- **Example:** Door slightly ajar opening slower; barely triggers motion detector

#### **Problem: Lighting changes mid-motion confuse the system**
**Use in combination with SSIM**
- **Increased intensity_threshold** + **Increased SSIM_threshold tolerance**
- **Why:** If lights change AND door opens, SSIM alone may not capture it; need intensity backup

### **4. Motion Threshold** – Optical Flow/Frame Difference

#### **Problem: Camera vibration/wind causes false motion detection**
**Increase motion threshold** (e.g., 3.0 → 4.0–5.0)
- **Conditions:** Outdoor camera, structural movement, camera shake
- **Why:** Reduces noise sensitivity; ignores small jitter
- **Example:** Tree branches moving, vibration from nearby traffic
- **Trade-off:** May miss very subtle door movement

#### **Problem: Door opens silently/slowly but motion is not detected**
**Decrease motion threshold** (e.g., 3.0 → 2.0–2.5)
- **Conditions:** Smooth slow door opening, hydraulic/silent closer
- **Why:** Threshold too high; gradual optical flow below it
- **Example:** Automatic door opener, soft-close mechanism
- **Risk:** Increased sensitivity to non-door motion

#### **Problem: People walking near door triggers false motion**
**Increase motion threshold** (e.g., 3.0 → 4.0)
- **Conditions:** DOOR_CORNER_ROI partially catches human movement
- **Why:** High threshold filters out fast human movement; only major changes count
- **Check:** Review ROI definitions; if possible, shrink DOOR_CORNER_ROI instead

### **Decision Matrix**

| **Scenario** | **Parameter** | **Adjust** | **Reason** |
|--|--|--|--|
| Door opens but not detected | `ssim_threshold` | ↑ Increase | Low-texture door, minimal visual change |
| False door-open alerts | `ssim_threshold` / `debounce` | ↓ Decrease / ↑ Increase | Shadows/reflections triggering SSIM dips |
| Slow/delayed detection | `debounce_threshold` | ↓ Decrease | Confirmation takes too long |
| Flickering door state | `ssim_threshold` + `debounce` | Widen margin + ↑ Debounce | SSIM hovering at threshold |
| Lights on/off false triggers | `intensity_threshold` | ↑ Increase | Bright jumps confusing door detector |
| Missed subtle door motion | `motion_threshold` | ↓ Decrease | Slow/quiet door opening |
| Camera vibration noise | `motion_threshold` | ↑ Increase | Environmental jitter creating false motion |
| People/traffic near door | `motion_threshold` | ↑ Increase | Non-door movement triggering detection |

### **Tuning Workflow**

1. **Start with baseline:** Use current config values
2. **Monitor logs:** Enable debug mode, capture SSIM/motion values during real events
3. **Identify pattern:** "Is it missing real openings?" vs. "False positives?"
4. **Adjust one param at a time:** Change only the most relevant parameter
5. **Test 50+ cycles:** Collect data over hours to filter out random noise
6. **Commit if stable:** Only lock values after 24+ hours of clean operation

*Developed for PMJ - Two-Man Rule Security Compliance*
