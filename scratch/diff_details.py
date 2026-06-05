import cv2
import numpy as np

ref = cv2.imread("scratch/patch_reference.png", cv2.IMREAD_GRAYSCALE)
p1380 = cv2.imread("scratch/patch_1380.png", cv2.IMREAD_GRAYSCALE)
p1404 = cv2.imread("scratch/patch_1404.png", cv2.IMREAD_GRAYSCALE)

ref_mean = np.mean(ref)
p1380_mean = np.mean(p1380)
p1404_mean = np.mean(p1404)

print(f"Ref mean: {ref_mean:.2f}")
print(f"1380 mean: {p1380_mean:.2f}, diff from ref: {abs(p1380_mean - ref_mean):.2f}")
print(f"1404 mean: {p1404_mean:.2f}, diff from ref: {abs(p1404_mean - ref_mean):.2f}")

diff_1380 = cv2.absdiff(p1380, ref)
diff_1404 = cv2.absdiff(p1404, ref)

print(f"Mean pixel diff 1380: {np.mean(diff_1380):.2f}")
print(f"Mean pixel diff 1404: {np.mean(diff_1404):.2f}")

# Brightness-normalized difference (same as in door_verifier)
offset_1380 = ref_mean - p1380_mean
p1380_adj = np.clip(p1380.astype(np.int16) + offset_1380, 0, 255).astype(np.uint8)
diff_adj_1380 = cv2.absdiff(p1380_adj, ref)

offset_1404 = ref_mean - p1404_mean
p1404_adj = np.clip(p1404.astype(np.int16) + offset_1404, 0, 255).astype(np.uint8)
diff_adj_1404 = cv2.absdiff(p1404_adj, ref)

print(f"Mean adjusted diff 1380: {np.mean(diff_adj_1380):.2f}")
print(f"Mean adjusted diff 1404: {np.mean(diff_adj_1404):.2f}")

# Find where the largest differences are in 1404
y_indices, x_indices = np.where(diff_adj_1404 > 15)
print(f"Number of pixels with diff > 15 in 1404: {len(y_indices)} out of {diff_adj_1404.size}")
if len(y_indices) > 0:
    print(f"Y range of large diffs: {np.min(y_indices)} to {np.max(y_indices)}")
    print(f"X range of large diffs: {np.min(x_indices)} to {np.max(x_indices)}")
