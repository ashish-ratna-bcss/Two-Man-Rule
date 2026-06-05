import csv

with open("scratch/raw_door_ssim_tirupathi.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

low_ssim_frames = []
for row in rows:
    f_idx = int(row["frame_idx"])
    ssim_val = float(row["ssim"])
    if ssim_val < 0.50:
        low_ssim_frames.append((f_idx, ssim_val))

print(f"Number of frames with SSIM < 0.50: {len(low_ssim_frames)}")
if low_ssim_frames:
    print("First 10 frames with SSIM < 0.50:")
    for f, s in low_ssim_frames[:10]:
        print(f"  Frame {f}: SSIM={s:.4f}")
    print("Last 10 frames with SSIM < 0.50:")
    for f, s in low_ssim_frames[-10:]:
        print(f"  Frame {f}: SSIM={s:.4f}")
