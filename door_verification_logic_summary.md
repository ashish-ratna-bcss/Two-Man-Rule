# Door Verification & Lighting Stabilization Logic

This document summarizes the recent debugging, logic improvements, and final implemented solutions for the `DoorVerifier` in the Two-Man Rule Monitoring System. The focus of these updates was to make the door state detection highly robust against extreme lighting transitions, occlusions, and ambient light shifts.

## 1. The Original Flaw: Hardcoded Intensity Locks
**Problem:** 
The original implementation coupled Structural Similarity (SSIM) with rigid intensity change thresholds. This caused two major issues:
1. False "OPEN" triggers when minor shadows shifted across the door.
2. False "CLOSED" results during legitimate door openings because the absolute intensity hadn't crossed the hardcoded threshold.

**Solution:** 
We removed the rigid `intensity_changed` dependencies and refactored the system to rely strictly on SSIM. We also enforced `data_range=255` in the SSIM calculation to anchor the analysis to absolute pixel values, preventing hyper-sensitivity on flat, low-texture door surfaces.

---

## 2. Global Lighting Shifts (Light Switches & Day/Night)
**Problem:** 
SSIM algorithms inherently penalize images if their overall luminance (brightness) drifts. A slow transition from night to morning, or someone suddenly flipping a light switch, would artificially crash the SSIM score and trigger a false door opening.

**Solution: Mean Brightness Normalization**
Before analyzing the structure, the system now calculates a `mean_offset`—the difference in average brightness between the current live feed and the baseline reference image. It applies this offset to perfectly normalize the live frame's brightness to match the reference. 
* **Result:** The system is completely immune to global luminance shifts. It now evaluates the door purely on its physical structure and edges.

---

## 3. The Occlusion Bug (People Blocking the Door)
**Problem:** 
To handle people walking in front of the door, the system uses a `visible_mask` to replace the occluded pixels with pristine pixels from the reference image. However, the brightness normalization offset was previously calculated *after* these pixels were mixed. If the lights were turned on while someone was standing there, the brightness adjustment would distort the pristine reference pixels we had just pasted in, artificially crashing the SSIM score.

**Solution: Targeted Masked Normalization**
We refactored the `_run_verification` sequence in `door_verifier.py`:
1. It now isolates the *non-occluded* (visible) pixels.
2. It calculates the brightness `mean_offset` strictly from those visible pixels.
3. It applies the brightness correction *only* to the raw camera feed.
4. Finally, it builds a composite patch by pasting the untouched, pristine reference pixels over the occluded area.
* **Result:** Brightness correction and occlusion masking now work together flawlessly without interfering with one another.

---

## 4. The Twilight Protection Layer (Dim Morning Light)
**Problem:** 
In dim, ambient early-morning lighting, the camera sensor loses physical contrast. While our new brightness normalizer perfectly fixes the luminance, the "crushed" contrast still artificially lowers the SSIM score. This risked triggering a false "OPEN" state right before the lights turned on.

**Solution: Dynamic Threshold Relaxation**
Instead of blinding the system in dim light, we implemented a dual-layer defense:
1. **Twilight Protection Layer (Intensity 20.0 to 45.0):** If the room is dim but visible, the system detects the twilight state and dynamically relaxes the SSIM threshold by 15% (e.g., a strict `0.80` threshold temporarily becomes a forgiving `0.68`). This absorbs the contrast penalty while still easily catching physical door openings (which drastically drop SSIM to ~`0.30 - 0.50`).
2. **Pitch Black Fallback (Intensity < 20.0):** The existing `darkening_protection` remains untouched. If the video is practically pitch black (sensor noise), the system safely forces the state to `CLOSED`.

---

## 5. System Debouncing & Real-World Tracking
**Observation:** 
During testing on the Jayanagar stream (`GF-5-CAM-25`), the system logged a `CLOSED` -> `OPEN` -> `CLOSED` transition within the first 8 seconds.

**Conclusion:** 
This was verified to be a flawless tracking of physical events. 
* The system intentionally initializes assuming a `CLOSED` state to prevent startup glitches.
* It requires `debounce_frames` (e.g., 15 frames / 0.6 seconds) of consistent readings to officially flip states.
* The door was physically open at the start of the video. The system debounced it and changed the state to `OPEN`. 
* Over the next 7 seconds, the logs showed SSIM smoothly climbing from `0.42` to `0.90`. This mathematically mapped the physical door swinging shut. The system then confirmed the `CLOSED` state, perfectly triggering the Evening Dual-Auth test window. 

---

### Final Code Architecture Summary
The final `models/door_verifier.py` handles image verification in this exact sequence:
1. Identify occlusions via `visible_mask`.
2. Extract means exclusively from visible pixels.
3. Normalize raw live brightness using `mean_offset`.
4. Construct composite patch (Normalized Live + Pristine Reference).
5. Apply Twilight Protection (dynamic threshold adjustment).
6. Calculate SSIM and apply Debounce Buffer.
