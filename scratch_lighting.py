import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Create reference (mean ~100, std ~30)
ref = np.full((100, 100), 100, dtype=np.uint8)
ref[20:80, 20:80] = 150
ref_mean = np.mean(ref)
ref_std = np.std(ref)

# Create dim current image (simulate ambient light)
# Divide by 2 (lower mean, lower contrast)
curr_dim = (ref.astype(np.float32) * 0.4).astype(np.uint8)
# Add some camera noise
noise = np.random.normal(0, 3, (100, 100)).astype(np.int8)
curr_dim = np.clip(curr_dim.astype(np.int16) + noise, 0, 255).astype(np.uint8)

curr_mean = np.mean(curr_dim)

# Method 1: Additive (Bias only)
offset = ref_mean - curr_mean
adj_add = np.clip(curr_dim.astype(np.int16) + offset, 0, 255).astype(np.uint8)

# Method 2: Multiplicative (Gain only)
gain = ref_mean / max(curr_mean, 1.0)
adj_mult = np.clip(curr_dim.astype(np.float32) * gain, 0, 255).astype(np.uint8)

ssim_raw = ssim(ref, curr_dim, full=False, data_range=255)
ssim_add = ssim(ref, adj_add, full=False, data_range=255)
ssim_mult = ssim(ref, adj_mult, full=False, data_range=255)

print(f"Ref  : mean={ref_mean:.1f}, std={ref_std:.1f}")
print(f"Curr : mean={curr_mean:.1f}, std={np.std(curr_dim):.1f}")
print(f"SSIM Raw: {ssim_raw:.3f}")
print(f"SSIM Additive: {ssim_add:.3f} (mean={np.mean(adj_add):.1f}, std={np.std(adj_add):.1f})")
print(f"SSIM Multiplicative: {ssim_mult:.3f} (mean={np.mean(adj_mult):.1f}, std={np.std(adj_mult):.1f})")

