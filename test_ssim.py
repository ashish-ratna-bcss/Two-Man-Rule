import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Create a low-texture patch (like a flat door)
ref = np.full((100, 100), 150, dtype=np.uint8)
# Add a little noise
noise = np.random.normal(0, 2, (100, 100)).astype(np.int8)
ref = np.clip(ref.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Create a shadow patch (drop intensity by 20)
curr = np.clip(ref.astype(np.int16) - 20, 0, 255).astype(np.uint8)

ssim_default = ssim(ref, curr, full=False)
ssim_255 = ssim(ref, curr, full=False, data_range=255)

print(f"Ref std: {np.std(ref):.2f}")
print(f"SSIM (default): {ssim_default:.3f}")
print(f"SSIM (data_range=255): {ssim_255:.3f}")

