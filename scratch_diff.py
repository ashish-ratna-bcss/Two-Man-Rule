import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

ref = np.full((100, 100), 100, dtype=np.uint8)
ref[20:80, 20:80] = 150
ref_mean = np.mean(ref)

curr_dim = (ref.astype(np.float32) * 0.4).astype(np.uint8)
noise = np.random.normal(0, 3, (100, 100)).astype(np.int8)
curr_dim = np.clip(curr_dim.astype(np.int16) + noise, 0, 255).astype(np.uint8)

curr_mean = np.mean(curr_dim)
offset = ref_mean - curr_mean
adj_add = np.clip(curr_dim.astype(np.int16) + offset, 0, 255).astype(np.uint8)

diff_raw = np.mean(cv2.absdiff(curr_dim, ref))
diff_adj = np.mean(cv2.absdiff(adj_add, ref))

print(f"Mean Diff (Raw): {diff_raw:.1f}")
print(f"Mean Diff (Adjusted): {diff_adj:.1f}")

