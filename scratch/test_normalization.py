import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_ssim(ref, curr):
    return ssim(ref, curr, full=False, data_range=255)

ref = cv2.imread("scratch/patch_reference.png", cv2.IMREAD_GRAYSCALE)
p0 = cv2.imread("scratch/patch_0.png", cv2.IMREAD_GRAYSCALE)
p1000 = cv2.imread("scratch/patch_1000.png", cv2.IMREAD_GRAYSCALE)
p1380 = cv2.imread("scratch/patch_1380.png", cv2.IMREAD_GRAYSCALE)
p1404 = cv2.imread("scratch/patch_1404.png", cv2.IMREAD_GRAYSCALE)

ref_mean = np.mean(ref)
ref_std = np.std(ref)

for name, patch in [("Frame 0", p0), ("Frame 1000", p1000), ("Frame 1380", p1380), ("Frame 1404", p1404)]:
    curr_mean = np.mean(patch)
    curr_std = np.std(patch)
    
    # Raw
    ssim_raw = compute_ssim(ref, patch)
    
    # Additive (current method)
    offset = ref_mean - curr_mean
    adj_add = np.clip(patch.astype(np.int16) + offset, 0, 255).astype(np.uint8)
    ssim_add = compute_ssim(ref, adj_add)
    
    # Multiplicative
    gain = ref_mean / max(curr_mean, 1.0)
    adj_mult = np.clip(patch.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    ssim_mult = compute_ssim(ref, adj_mult)
    
    # Standard Score (Mean + Std Dev normalization)
    if curr_std > 0.1:
        adj_std = (patch.astype(np.float32) - curr_mean) * (ref_std / curr_std) + ref_mean
        adj_std = np.clip(adj_std, 0, 255).astype(np.uint8)
        ssim_std = compute_ssim(ref, adj_std)
    else:
        ssim_std = 0.0
        
    print(f"{name}:")
    print(f"  Mean: {curr_mean:.2f} (Ref: {ref_mean:.2f}) | Std: {curr_std:.2f} (Ref: {ref_std:.2f})")
    print(f"  SSIM Raw           : {ssim_raw:.4f}")
    print(f"  SSIM Additive      : {ssim_add:.4f}")
    print(f"  SSIM Multiplicative: {ssim_mult:.4f}")
    print(f"  SSIM Mean+Std      : {ssim_std:.4f}")
