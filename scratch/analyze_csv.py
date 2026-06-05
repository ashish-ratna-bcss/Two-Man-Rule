import csv

with open("scratch/raw_door_ssim_tirupathi.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("Total frames processed:", len(rows))

drops = []
current_drop_start = None
current_min_ssim = 1.0

for i, row in enumerate(rows):
    f_idx = int(row["frame_idx"])
    ssim_val = float(row["ssim"])
    
    if ssim_val < 0.80:
        if current_drop_start is None:
            current_drop_start = f_idx
            current_min_ssim = ssim_val
        else:
            current_min_ssim = min(current_min_ssim, ssim_val)
    else:
        if current_drop_start is not None:
            drops.append((current_drop_start, f_idx - 1, current_min_ssim))
            current_drop_start = None

if current_drop_start is not None:
    drops.append((current_drop_start, len(rows) - 1, current_min_ssim))

print("SSIM Drop segments below 0.80:")
for start_f, end_f, min_s in drops:
    duration_frames = end_f - start_f + 1
    duration_secs = duration_frames / 25.0
    print(f"Frames {start_f} to {end_f} ({duration_frames} frames, {duration_secs:.2f}s) - min SSIM: {min_s:.4f}")

# Print the SSIM around the transition
print("\nSample SSIM values around frame 1380-1410:")
for idx in range(1370, 1420):
    if idx < len(rows):
        row = rows[idx]
        print(f"Frame {row['frame_idx']}: SSIM={float(row['ssim']):.4f}, mean_diff={float(row['mean_diff']):.1f}, stable={row['stable']}")
        
# Find the frame where the first stable transition to OPEN occurs
stable_open_frames = [int(row["frame_idx"]) for row in rows if row["stable"] == "True"]
if stable_open_frames:
    print(f"\nFirst frame where stable state becomes OPEN: {stable_open_frames[0]}")
else:
    print("\nStable state never becomes OPEN in this CSV.")
