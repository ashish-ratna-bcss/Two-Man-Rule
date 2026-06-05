import cv2
import numpy as np
import csv
from datetime import datetime, timedelta, timezone
from models.door_verifier import DoorVerifier
from config import STREAMS_CONFIG

# Stream 15 is Tirupathi
stream_config = STREAMS_CONFIG[15]
print("Loaded Stream Config for:", stream_config["site_name"])

door_verifier = DoorVerifier(
    stream_config["closed_door_reference"],
    door_corner_roi=stream_config["rois"]["DOOR_CORNER_ROI"],
    similarity_threshold=stream_config["ssim_threshold"],
    debounce_threshold=stream_config["debounce_threshold"],
    intensity_threshold=stream_config["intensity_threshold"],
    motion_threshold=stream_config["motion_threshold"],
    darkening_protection=stream_config.get("darkening_protection", True),
    min_visible_ratio=stream_config.get("door_corner_min_visible_ratio", 0.5),
)

cap = cv2.VideoCapture("GF-23-14-M.mp4")
frame_idx = 0
data = []

start_time = datetime(2026, 6, 5, 10, 23, 26, tzinfo=timezone(timedelta(hours=5, minutes=30)))

min_ssim = 1.0
max_mean_diff = 0.0

with open("scratch/raw_door_ssim_tirupathi.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_idx", "ssim", "curr_mean", "mean_diff", "is_open", "candidate", "stable"])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        ts_ist = start_time + timedelta(seconds=frame_idx / 25.0)
        
        is_open = door_verifier.verify(frame, tracked_persons=None, ts_ist=ts_ist)
        
        ssim_val = door_verifier.get_last_ssim()
        curr_mean = door_verifier.get_last_intensity()
        mean_diff = door_verifier.get_last_mean_diff()
        
        if ssim_val is not None:
            min_ssim = min(min_ssim, ssim_val)
        if mean_diff is not None:
            max_mean_diff = max(max_mean_diff, mean_diff)
            
        writer.writerow([
            frame_idx,
            ssim_val,
            curr_mean,
            mean_diff,
            is_open,
            door_verifier.candidate_state,
            door_verifier.stable_is_open
        ])
        
        frame_idx += 1

cap.release()

print("Finished. Saved to scratch/raw_door_ssim_tirupathi.csv")
print("Min SSIM:", min_ssim)
print("Max mean_diff:", max_mean_diff)
