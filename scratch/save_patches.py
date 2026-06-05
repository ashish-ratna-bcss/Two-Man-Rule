import cv2
import numpy as np
from config import STREAMS_CONFIG

stream_config = STREAMS_CONFIG[15]
roi = stream_config["rois"]["DOOR_CORNER_ROI"]

# Bounding box of ROI
rx = int(np.min(roi[:, 0]))
ry = int(np.min(roi[:, 1]))
rw = int(np.max(roi[:, 0]) - rx)
rh = int(np.max(roi[:, 1]) - ry)

print(f"ROI Bounding Box: rx={rx}, ry={ry}, rw={rw}, rh={rh}")

cap = cv2.VideoCapture("GF-23-14-M.mp4")

# Load reference image
ref_img = cv2.imread(stream_config["closed_door_reference"])
ref_crop = ref_img[ry:ry+rh, rx:rx+rw]
cv2.imwrite("scratch/patch_reference.png", ref_crop)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx in [0, 1000, 1380, 1404]:
        crop = frame[ry:ry+rh, rx:rx+rw]
        cv2.imwrite(f"scratch/patch_{frame_idx}.png", crop)
        print(f"Saved patch for frame {frame_idx}")
        
    frame_idx += 1

cap.release()
