import cv2
import numpy as np
import csv
from datetime import datetime, timedelta, timezone
from models.door_verifier import DoorVerifier
from config import STREAMS_CONFIG

# Stream 15 is Tirupathi
stream_config = STREAMS_CONFIG[15]
print("Loaded Stream Config for:", stream_config["site_name"])

class DoorVerifierUpward(DoorVerifier):
    def _build_visible_mask(self, tracked_persons) -> np.ndarray:
        occlusion_mask = np.zeros_like(self._roi_mask, dtype=np.uint8)
        if tracked_persons:
            scale_x = self.ssim_size[0] / max(self.rw, 1)
            scale_y = self.ssim_size[1] / max(self.rh, 1)
            for person in tracked_persons.values():
                bbox = person.get("bbox")
                if bbox is None or len(bbox) < 4:
                    continue
                
                # Original bbox coords
                bx1, by1, bx2, by2 = bbox[0], bbox[1], bbox[2], bbox[3]
                h = by2 - by1
                
                # Extend top of bbox upwards by 35% of height
                by1_extended = by1 - 0.35 * h
                
                x1 = max(float(bx1), float(self.rx))
                y1 = max(float(by1_extended), float(self.ry))
                x2 = min(float(bx2), float(self.rx + self.rw))
                y2 = min(float(by2), float(self.ry + self.rh))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                lx1 = int(np.floor((x1 - self.rx) * scale_x))
                ly1 = int(np.floor((y1 - self.ry) * scale_y))
                lx2 = int(np.ceil((x2 - self.rx) * scale_x))
                ly2 = int(np.ceil((y2 - self.ry) * scale_y))
                
                lx1 = max(0, min(lx1, self.ssim_size[0] - 1))
                ly1 = max(0, min(ly1, self.ssim_size[1] - 1))
                lx2 = max(0, min(lx2, self.ssim_size[0]))
                ly2 = max(0, min(ly2, self.ssim_size[1]))
                
                if lx2 <= lx1 or ly2 <= ly1:
                    continue
                
                cv2.rectangle(occlusion_mask, (lx1, ly1), (lx2 - 1, ly2 - 1), 1, thickness=-1)
                
        visible_mask = np.where((self._roi_mask == 1) & (occlusion_mask == 0), 1, 0).astype(np.uint8)
        visible_pixels = int(np.count_nonzero(visible_mask))
        self.last_visible_ratio = (visible_pixels / self._roi_area_pixels) if self._roi_area_pixels else 0.0
        return visible_mask

# We'll use the PoseDetector and PersonTracker to track people
from models.pose_detector import PoseDetector
from models.tracker import PersonTracker

detector = PoseDetector()
tracker = PersonTracker()

door_verifier = DoorVerifierUpward(
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

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    ts_ist = start_time + timedelta(seconds=frame_idx / 25.0)
    
    detections = detector.detect(frame)
    tracked_persons = tracker.update(detections) if detections is not None else {}
    
    is_open = door_verifier.verify(frame, tracked_persons=tracked_persons, ts_ist=ts_ist)
    
    ssim_val = door_verifier.get_last_ssim()
    curr_mean = door_verifier.get_last_intensity()
    mean_diff = door_verifier.get_last_mean_diff()
    vis_ratio = door_verifier.last_visible_ratio
    
    data.append({
        "frame_idx": frame_idx,
        "ssim": ssim_val,
        "curr_mean": curr_mean,
        "mean_diff": mean_diff,
        "visible_ratio": vis_ratio,
        "is_open": is_open,
        "candidate": door_verifier.candidate_state,
        "stable": door_verifier.stable_is_open
    })
    
    # Print progress every 100 frames
    if frame_idx % 100 == 0 or (is_open and frame_idx < 1550):
        print(f"Frame {frame_idx}: SSIM={ssim_val:.3f}, VisRatio={vis_ratio:.3f}, Candidate={door_verifier.candidate_state}, Stable={door_verifier.stable_is_open}")
        
    frame_idx += 1

cap.release()

with open("scratch/upward_door_ssim_tirupathi.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_idx", "ssim", "curr_mean", "mean_diff", "visible_ratio", "is_open", "candidate", "stable"])
    for row in data:
        writer.writerow([row["frame_idx"], row["ssim"], row["curr_mean"], row["mean_diff"], row["visible_ratio"], row["is_open"], row["candidate"], row["stable"]])

print("Saved to scratch/upward_door_ssim_tirupathi.csv")
