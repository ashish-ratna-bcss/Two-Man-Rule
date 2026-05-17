import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Base reference
ref = np.full((100, 100), 100, dtype=np.uint8)
# Add structure
ref[40:60, 40:60] = 150
noise = np.random.normal(0, 2, (100, 100)).astype(np.int8)
ref = np.clip(ref.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Lights ON (intensity increases by 60)
curr = np.clip(ref.astype(np.int16) + 60, 0, 255).astype(np.uint8)

ssim_standard = ssim(ref, curr, full=False, data_range=255)

ref_mean = np.mean(ref)
curr_mean = np.mean(curr)
mean_offset = ref_mean - curr_mean
adjusted_curr = np.clip(curr.astype(np.int16) + mean_offset, 0, 255).astype(np.uint8)

ssim_adjusted = ssim(ref, adjusted_curr, full=False, data_range=255)

print(f"SSIM (standard lights ON): {ssim_standard:.3f}")
print(f"SSIM (mean adjusted lights ON): {ssim_adjusted:.3f}")

