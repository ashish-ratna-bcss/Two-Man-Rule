import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from config import STREAMS_CONFIG

stream_config = STREAMS_CONFIG[15]
roi = stream_config["rois"]["DOOR_CORNER_ROI"]

# Bounding box of ROI
rx = int(np.min(roi[:, 0]))
ry = int(np.min(roi[:, 1]))
rw = int(np.max(roi[:, 0]) - rx)
rh = int(np.max(roi[:, 1]) - ry)

ssim_size = (100, 33) # calculated size from DoorVerifier

# Resize ROI polygon to ssim_size
local_polygon = roi.astype(np.float32)
local_polygon[:, 0] = (local_polygon[:, 0] - rx) * ssim_size[0] / max(rw, 1)
local_polygon[:, 1] = (local_polygon[:, 1] - ry) * ssim_size[1] / max(rh, 1)
local_polygon = np.rint(local_polygon).astype(np.int32)
roi_mask = np.zeros((ssim_size[1], ssim_size[0]), dtype=np.uint8)
cv2.fillPoly(roi_mask, [local_polygon], 1)

ref_img = cv2.imread(stream_config["closed_door_reference"], cv2.IMREAD_GRAYSCALE)
ref_crop = ref_img[ry:ry+rh, rx:rx+rw]
reference_patch = cv2.resize(ref_crop, ssim_size)

p0 = cv2.imread("scratch/patch_0.png", cv2.IMREAD_GRAYSCALE)
p1000 = cv2.imread("scratch/patch_1000.png", cv2.IMREAD_GRAYSCALE)
p1380 = cv2.imread("scratch/patch_1380.png", cv2.IMREAD_GRAYSCALE)
p1404 = cv2.imread("scratch/patch_1404.png", cv2.IMREAD_GRAYSCALE)

def verify_method(patch, method_type):
    curr_patch = cv2.resize(patch, ssim_size)
    visible_mask = roi_mask.copy()
    
    visible_curr = curr_patch[visible_mask == 1]
    visible_ref = reference_patch[visible_mask == 1]
    
    ref_mean = np.mean(visible_ref)
    curr_mean = np.mean(visible_curr)
    
    if method_type == "raw":
        adjusted_curr = curr_patch
    elif method_type == "add":
        offset = ref_mean - curr_mean
        adjusted_curr = np.clip(curr_patch.astype(np.int16) + offset, 0, 255).astype(np.uint8)
    elif method_type == "mult":
        gain = ref_mean / max(curr_mean, 1.0)
        adjusted_curr = np.clip(curr_patch.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    elif method_type == "std":
        curr_std = np.std(visible_curr)
        ref_std = np.std(visible_ref)
        if curr_std > 0.1:
            adjusted_curr = (curr_patch.astype(np.float32) - curr_mean) * (ref_std / curr_std) + ref_mean
            adjusted_curr = np.clip(adjusted_curr, 0, 255).astype(np.uint8)
        else:
            adjusted_curr = curr_patch
            
    composite_patch = reference_patch.copy()
    composite_patch[visible_mask == 1] = adjusted_curr[visible_mask == 1]
    
    val = float(ssim(reference_patch, composite_patch, full=False, data_range=255))
    return val

for name, patch in [("Frame 0", p0), ("Frame 1000", p1000), ("Frame 1380", p1380), ("Frame 1404", p1404)]:
    print(f"{name}:")
    print(f"  SSIM Raw           : {verify_method(patch, 'raw'):.4f}")
    print(f"  SSIM Additive      : {verify_method(patch, 'add'):.4f}")
    print(f"  SSIM Multiplicative: {verify_method(patch, 'mult'):.4f}")
    print(f"  SSIM Mean+Std      : {verify_method(patch, 'std'):.4f}")
