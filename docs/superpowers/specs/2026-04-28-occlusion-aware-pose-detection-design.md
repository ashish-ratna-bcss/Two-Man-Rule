# Occlusion-Aware Dual-Person ID Assignment & Tracking

**Date:** 2026-04-28  
**Scope:** Extend DualAuthStateMachine to maintain ID assignments and tracking when either person's ankles become occluded during unlock interaction.

## Problem

Current pose qualification requires full visibility: head in INTERACTION_ZONE, feet in STANDING_ZONE, both lock contacts, correct arm geometry.

When Person 1 steps aside to give room to Person 2, Person 1 may become partially occluded (ankles hidden by Person 2's body or furniture). Without fallback rules, Person 1's timer resets, breaking the unlock sequence.

## Requirements

1. **ID assignment gate:** Only normal pose (full visibility) triggers initial ID assignment (1 or 2)
2. **Parallel tracking:** Both id_a and id_b tracked simultaneously once assigned
3. **Occlusion recovery:** When person's ankles drop below 0.3 confidence AND person already has ID assigned, fallback rules maintain qualification
4. **Fallback rules:** Elbows left/right correct + head near LOCKER_A polygon + both lock contacts present
5. **Persistence:** IDs persist until door opens; no re-assignment

## Design

### Phase 1: Normal Pose Qualification (ID Assignment)

**Trigger:** `_evaluate_unlock_pose()` returns `qualified=true`

**Conditions:**
- Head in INTERACTION_ZONE
- Feet (ankle) in STANDING_ZONE (ankle confidence ≥ threshold)
- Both lock contacts (LOCK_A_ROI + LOCK_B_ROI)
- Elbows left/right in correct video order
- Arms raised toward door

**Action:** Assign id_a or id_b; store spatial anchor; start unlock timer

**No changes to existing logic here.** This gate remains strict.

### Phase 2: Parallel Tracking & Fallback Detection

**When updating timers for persons with assigned IDs:**

1. Run normal `_evaluate_unlock_pose()` for all persons
2. For each person with id_a or id_b:
   - If normal qualification is false AND both ankles < 0.3 confidence:
     - Run `_check_occlusion_fallback(person, track_id)`
     - If fallback passes: override qualification to true, mark `occlusion_mode=true`
3. Continue timer updates with "qualified" status (normal or fallback-rescued)
4. Anchor-based track remapping (existing `_find_matching_track()`) runs regardless of pose mode

### Phase 3: Fallback Pose Validation

**Method:** `_check_occlusion_fallback(person, track_id) -> bool`

**Conditions:**
- Both ankles < 0.3 confidence (occlusion detected)
- Elbows left/right in correct video order (same as normal pose)
- Head within minimum distance to LOCKER_A polygon boundary
- Both lock contacts (LOCK_A + LOCK_B) still present
- Anchor distance to stored id_a/id_b anchor ≤ UNLOCKER_ANCHOR_MATCH_PIXELS

**Returns:** True if all pass; False otherwise

**Note:** Fallback does NOT require feet in STANDING_ZONE. It allows person to move away while maintaining timer.

## Implementation

### Code Changes in `logic/state_machine.py`

**New constant (add to config.py):**
```python
ANKLE_OCCLUSION_CONFIDENCE_THRESHOLD = 0.3
```

**New method in DualAuthStateMachine:**
```python
def _check_occlusion_fallback(self, person: Dict, track_id: int) -> bool:
    """
    Fallback qualification when ankles occluded.
    Used only for persons already assigned id_a or id_b.
    """
    keypoints = person.get("keypoints")
    if keypoints is None:
        return False
    
    # Verify ankle occlusion
    ar_conf = keypoints[config.KEYPOINT_ANKLE_RIGHT][2]
    al_conf = keypoints[config.KEYPOINT_ANKLE_LEFT][2]
    if not (ar_conf < config.ANKLE_OCCLUSION_CONFIDENCE_THRESHOLD and 
            al_conf < config.ANKLE_OCCLUSION_CONFIDENCE_THRESHOLD):
        return False
    
    # Check elbow order
    if not self._left_right_keypoints_in_video_order(keypoints):
        return False
    
    # Check head near LOCKER_A
    head_pos = self._get_head_position(keypoints)
    if head_pos is None:
        return False
    
    locker_a_center = self.roi_manager.get_roi_center("LOCKER_A_ROI")
    if locker_a_center is None:
        return False
    
    # Minimum distance to LOCKER_A polygon
    min_dist = self._min_distance_to_polygon("LOCKER_A_ROI", head_pos)
    if min_dist > config.HEAD_TO_LOCKER_A_MAX_PIXELS:  # TBD: threshold
        return False
    
    # Check both lock contacts
    if not (self._side_contacts_any_lock(keypoints, "left") and
            self._side_contacts_any_lock(keypoints, "right")):
        return False
    
    return True
```

**Modify `update_timers()` in DualAuthStateMachine:**

After line 118 (`self._refresh_verified_slots(pose_results)`), add:

```python
# Check fallback qualification for assigned persons
self._apply_occlusion_recovery(tracked_persons, pose_results)
```

**New method:**
```python
def _apply_occlusion_recovery(self, tracked_persons: Dict, pose_results: Dict):
    """
    Override qualification to true via fallback rules if person is assigned
    but normal pose qualification failed due to ankle occlusion.
    """
    for slot in ("a", "b"):
        id_key = f"id_{slot}"
        assigned_id = self.session.get(id_key)
        if assigned_id is None:
            continue
        
        if assigned_id not in pose_results:
            continue
        
        pose = pose_results[assigned_id]
        if pose["qualified"]:
            continue  # Already qualified normally
        
        person = tracked_persons.get(assigned_id)
        if person is None:
            continue
        
        if self._check_occlusion_fallback(person, assigned_id):
            pose_results[assigned_id]["qualified"] = True
            pose_results[assigned_id]["occlusion_mode"] = True
            print(f"[OCCLUSION] P{1 if slot == 'a' else 2} fallback qualified")
```

### New Config Constants

Add to `config.py`:
```python
ANKLE_OCCLUSION_CONFIDENCE_THRESHOLD = 0.3  # Both ankles below this = occlusion
HEAD_TO_LOCKER_A_MAX_PIXELS = 150  # Max distance from head to LOCKER_A polygon
```

### New ROI Helper (if needed)

If `_min_distance_to_polygon()` doesn't exist, add to `logic/state_machine.py`:
```python
def _min_distance_to_polygon(self, roi_name: str, point: Tuple[float, float]) -> float:
    """
    Minimum distance from point to any edge of ROI polygon.
    """
    # Fetch polygon points from roi_manager
    # Calculate min distance to all polygon edges
    # Return minimum
    # (Requires roi_manager to expose polygon vertices)
```

## Testing

- **Unit:** Test `_check_occlusion_fallback()` with synthetic keypoints (ankles < 0.3, elbows correct, head near locker)
- **Integration:** Video with two persons; Person 1 steps aside, Person 2 overlaps. Verify id_a continues timer despite ankle occlusion
- **Edge cases:**
  - Person steps out of frame (track_id lost) → anchor remapping should recover
  - Both persons' ankles occluded simultaneously → both use fallback
  - Ankles reappear mid-unlock → normal qualification resumes

## Rollback / Feature Flag

If needed, fallback can be disabled by always returning false from `_check_occlusion_fallback()` or skipping `_apply_occlusion_recovery()` call.

---

**Status:** Ready for implementation review
