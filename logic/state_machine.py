import math
from typing import Dict, Optional, Set, Tuple

import numpy as np

import config
from logic.roi_manager import ROIManager


HEAD_KEYPOINTS = (0, 1, 2, 3, 4)



class DualAuthStateMachine:
    """
    Sequential dual-custody unlock state machine.

    Only a person showing the full unlock pose is promoted into this flow:
    head inside INTERACTION_ZONE, base inside STANDING_ZONE, both arms raised toward
    the lock area, and hand/elbow contact with both lock ROIs. Everyone else remains
    a raw tracker detection and is ignored by the authorization logic.
    """

    def __init__(self, roi_manager: ROIManager, fps: int = 30, session_id: Optional[str] = None,
                 mirror_left_right: bool = False,
                 min_unlock_seconds: float = None, max_unlock_seconds: float = None):
        self.roi_manager = roi_manager
        self.fps = max(int(fps or config.DEFAULT_FPS), 1)
        _min_s = float(min_unlock_seconds) if min_unlock_seconds is not None else config.MIN_UNLOCK_SECONDS
        _max_s = float(max_unlock_seconds) if max_unlock_seconds is not None else config.MAX_UNLOCK_SECONDS
        self.min_unlock_frames = max(1, int(_min_s * self.fps))
        self.max_unlock_frames = max(self.min_unlock_frames + 1, int(_max_s * self.fps))
        self.dwell_frames = self.min_unlock_frames
        self.session = config.create_session()
        self.mirror_left_right = mirror_left_right  # True for top-down cameras where L/R appears flipped

        self.active_ids_in_zone: Set[int] = set()
        self.current_frame_count = 0

        # Clearance gate state for door verification.
        self.standing_zone_was_occupied = False
        self.clearance_frames_since_empty = 0
        self.clearance_buffer_frames = 3
        self.door_roi_person_overlap = False

        # Saved once both different unlockers complete their 6-10s interaction.
        # Stable physical-person anchors. These let P1/P2 survive raw tracker ID churn.
        self.slot_anchors = {"a": None, "b": None}
        self.verified_anchors = {"a": None, "b": None}

        # Unlocker tagging: multiple IDs per person with same tag
        # unlocker_tags[track_id] = "P1_unlocker" or "P2_unlocker"
        # all_unlocker_ids["P1_unlocker"] = {main_id, alt_id1, alt_id2, ...}
        self.unlocker_tags = {}  # track_id -> "P1_unlocker" or "P2_unlocker"
        self.all_unlocker_ids = {"P1_unlocker": set(), "P2_unlocker": set()}

        # Body fingerprints saved at qualification: slot -> {height, width, pose_emb}
        # Used for multi-factor ReID to prevent wrong-person swaps
        self.body_fingerprints = {"a": None, "b": None}

        # Last known bbox per slot — updated every frame the person is directly detected
        # Used for live position tracking (search origin for ReID)
        self.last_seen_bbox = {"a": None, "b": None}

        # Bbox frozen at verification time — used as height reference in ReID size gate.
        # last_seen_bbox can shrink to a partial edge-clip when person goes aside;
        # using that as height ref would wrongly reject a full-body re-entry.
        self.slot_height_ref = {"a": None, "b": None}

        # Bbox at the moment the candidate is first picked — used as height reference in
        # candidate-level _find_matching_track to avoid picking a wrong nearby person.
        self.candidate_bbox = {"a": None, "b": None}

        # Last known head (x, y) per slot — fallback for interaction-zone check during synthetic hold
        self.last_seen_head_pos = {"a": None, "b": None}

        # Frames since each verified slot was last directly seen — drives dynamic remap radius
        self.slot_lost_frames = {"a": 0, "b": 0}

        # Departure tracking: frames since verified unlocker last found in tracked_persons (any pose).
        # Once slot_departed[slot] = True it is irreversible for this session — a different physical
        # person inheriting the same tracker ID cannot satisfy the zone presence check.
        self.slot_departed = {"a": False, "b": False}
        self.slot_zone_absent_frames = {"a": 0, "b": 0}

        # Per-person sequential lock interaction progress.
        # A tracker ID is only promoted to candidate AFTER
        # all 4 arm keypoints (LW, LE, RW, RL) are inside LOCKS_ROI simultaneously.

    # ================================================================
    # CENSUS / CLEARANCE
    # ================================================================
    def update_occupancy(self, tracked_persons: Dict[int, Dict], frame_step: int = 1):
        """
        Update only physical room/door clearance.

        Authorization occupancy is intentionally not based on everyone in the room.
        The assigned IDs are chosen later, only from people performing the unlock
        pose with their head inside INTERACTION_ZONE.
        """
        frame_step = max(int(frame_step), 1)
        self.current_frame_count += frame_step
        self.active_ids_in_zone = set()
        standing_zone_occupied = False
        door_roi_overlap = False

        for person in tracked_persons.values():
            bbox = person.get("bbox")
            keypoints = person.get("keypoints")
            if bbox is None:
                continue

            if keypoints is not None and len(keypoints) > config.KEYPOINT_HIP_RIGHT:
                fx, fy = self._get_base_position(keypoints, bbox)
            else:
                fx, fy = self._bbox_bottom_center(bbox)

            if self.roi_manager.point_in_roi("STANDING_ZONE", fx, fy):
                standing_zone_occupied = True

            if self._bbox_overlaps_door_roi(bbox):
                door_roi_overlap = True

        self.door_roi_person_overlap = door_roi_overlap
        self._update_clearance_state(standing_zone_occupied, frame_step)
        return "OK"

    def _update_clearance_state(self, standing_zone_occupied: bool, frame_step: int = 1):
        if standing_zone_occupied:
            self.standing_zone_was_occupied = True
            self.clearance_frames_since_empty = 0
        else:
            self.clearance_frames_since_empty += frame_step

    def should_check_door_state(self) -> bool:
        """
        Observe -> wait for clear -> verify.

        The door image is checked only after someone has used the standing zone,
        the zone has gone empty for a few frames, and no bbox is occluding DOOR_ROI.
        """
        return (
            self.standing_zone_was_occupied
            and self.clearance_frames_since_empty >= self.clearance_buffer_frames
            and not self.door_roi_person_overlap
        )

    # ================================================================
    # SEQUENTIAL UNLOCK TIMERS
    # ================================================================
    def update_timers(self, tracked_persons: Dict[int, Dict], frame_step: int = 1):
        frame_step = max(int(frame_step), 1)
        pose_results = {}
        interacting_ids: Set[int] = set()


        zone_occupied = False
        for track_id, person in tracked_persons.items():
            pose = self._evaluate_unlock_pose(person, track_id)
            pose_results[track_id] = pose
            
            # Active (Priority) Trigger:
            # 1. MUST be standing in STANDING_ZONE
            # 2. MUST be facing the door (shoulders or general body order must be correct)
            is_standing = pose.get("feet_in_standing", False)
            facing_correct = pose.get("shoulder_order_correct", False) or pose.get("left_right_order", False)
            
            if is_standing and facing_correct:
                zone_occupied = True
                
            if pose["qualified"]:
                interacting_ids.add(track_id)
                
        self.session["zone_occupied"] = zone_occupied

        self._apply_occlusion_recovery(tracked_persons, pose_results)

        for track_id, pose in pose_results.items():
            if pose["qualified"]:
                interacting_ids.add(track_id)

        # Once assigned, keep person alive - add to interacting_ids if available, else use grace buffer
        for slot in ("a", "b"):
            assigned_id = self.session.get(f"id_{slot}")
            candidate_id = self.session.get(f"candidate_{slot}")
            timer = self.session.get(f"timer_{slot}_frames", 0)

            # Persistent slot tracking: increment lost frames regardless of whether ID is currently assigned.
            # This allows us to track how long a verified person has been missing.
            if assigned_id is None and self.verified_anchors[slot] is not None:
                self.slot_lost_frames[slot] += 1
                # Attempt recovery for empty verified slot
                remapped = self._remap_verified_unlocker(slot, tracked_persons, pose_results)
                if remapped:
                    assigned_id = self.session[f"id_{slot}"]
                    self.slot_lost_frames[slot] = 0
                    interacting_ids.add(assigned_id)
                    # Recovery successful! Continue to the next slot or process normally.
                else:
                    # Still missing — continue to next slot
                    continue

            if assigned_id is not None:
                if assigned_id in pose_results:
                    self.slot_lost_frames[slot] = 0  # reset — directly visible this frame
                    interacting_ids.add(assigned_id)
                    if assigned_id in tracked_persons:
                        bbox = tracked_persons[assigned_id].get("bbox")
                        if bbox is not None:
                            self.last_seen_bbox[slot] = bbox
                        kpts = tracked_persons[assigned_id].get("keypoints")
                        if kpts is not None:
                            hp = self._get_head_position(kpts)
                            if hp is not None:
                                self.last_seen_head_pos[slot] = hp
                else:
                    self.slot_lost_frames[slot] += 1
                    # ID lost — attempt strict state-machine ReID before synthetic hold
                    remapped = self._remap_verified_unlocker(slot, tracked_persons, pose_results)
                    assigned_id = self.session[f"id_{slot}"]  # re-read in case remapped

                    if remapped and assigned_id in pose_results:
                        self.slot_lost_frames[slot] = 0
                        interacting_ids.add(assigned_id)
                        if assigned_id in tracked_persons:
                            bbox = tracked_persons[assigned_id].get("bbox")
                            if bbox is not None:
                                self.last_seen_bbox[slot] = bbox
                    else:
                        # No spatial match — synthetic hold until tracker recovers or timeout
                        if self.slot_lost_frames[slot] > config.MAX_SYNTHETIC_HOLD_FRAMES:
                            print(f"[ASSIGN] P{1 if slot == 'a' else 2} ID {assigned_id} lost for too long "
                                  f"({self.slot_lost_frames[slot]}f). Suspending verified ID.")
                            self.session[f"id_{slot}"] = None
                            # DO NOT clear verified_anchors or body_fingerprints — we need them for ReID recovery!
                            # self.verified_anchors[slot] = None 
                            # self.slot_lost_frames[slot] = 0 # Keep the count going for ReID radius expansion
                            
                            # Reset sequence state appropriately
                            if slot == "a":
                                self.session["sequence_state"] = "WAITING_FOR_FIRST_UNLOCKER"
                            else:
                                if self.session.get("id_a") is not None:
                                    self.session["sequence_state"] = "WAITING_FOR_SECOND_UNLOCKER"
                                else:
                                    self.session["sequence_state"] = "WAITING_FOR_FIRST_UNLOCKER"
                            continue

                        print(f"[ASSIGN] P{1 if slot == 'a' else 2} ID {assigned_id} not in tracker, "
                              f"synthetic hold (lost {self.slot_lost_frames[slot]}f)")
                        pose_results[assigned_id] = {
                            "qualified": True,
                            "has_lock_contact": False,
                            "head_in_interaction": True,
                            "head_in_door": False,
                            "shoulders_in_door": False,
                            "shoulder_order_correct": True,
                            "feet_in_standing": True,
                            "anchor": self.verified_anchors[slot]
                        }
                        interacting_ids.add(assigned_id)
                continue

            # Candidate: only keep in interacting_ids if pose STILL qualifies this frame.
            # Dropping them here activates the grace buffer in _update_unlock_slot so
            # brief single-frame lapses are tolerated; sustained pose failure resets timer.
            if candidate_id is not None and timer > 0 and candidate_id in tracked_persons:
                if pose_results.get(candidate_id, {}).get("qualified", False):
                    interacting_ids.add(candidate_id)
                    print(f"[CANDI] P{1 if slot == 'a' else 2} candidate ID {candidate_id} (timer={timer}f)")

        self._refresh_verified_slots(pose_results)

        # ── Departure detection ───────────────────────────────────────────────
        # Track whether each verified unlocker is STILL the same physical person.
        # slot_departed is set irreversibly when:
        #   (a) The assigned ID disappears from tracked_persons for >= DEPARTURE_FRAMES_THRESHOLD, OR
        #   (b) The bbox height of whoever now holds that ID mismatches the reference by >40%
        #       (indicates a different person inherited the tracker ID via ReID alias).
        for slot in ("a", "b"):
            if self.slot_departed[slot]:
                continue
            assigned_id = self.session.get(f"id_{slot}")
            if assigned_id is None or self.verified_anchors[slot] is None:
                continue

            if assigned_id in tracked_persons:
                # Check height continuity — catch ByteTrack ReID alias to a different person
                height_ref_bbox = self.slot_height_ref.get(slot) or self.last_seen_bbox.get(slot)
                if height_ref_bbox is not None:
                    ref_h = float(height_ref_bbox[3] - height_ref_bbox[1])
                    cand_bbox = tracked_persons[assigned_id].get("bbox")
                    if cand_bbox is not None and ref_h > 0:
                        cand_h = float(cand_bbox[3] - cand_bbox[1])
                        if abs(cand_h - ref_h) / ref_h > 0.4:
                            print(f"[DEPART] P{1 if slot == 'a' else 2} ID {assigned_id} "
                                  f"height mismatch ref={ref_h:.0f}px cand={cand_h:.0f}px → "
                                  f"different person inherited ID → departed")
                            self.slot_departed[slot] = True
                            continue
                self.slot_zone_absent_frames[slot] = 0
            else:
                self.slot_zone_absent_frames[slot] += frame_step
                if self.slot_zone_absent_frames[slot] >= config.DEPARTURE_FRAMES_THRESHOLD:
                    print(f"[DEPART] P{1 if slot == 'a' else 2} ID {assigned_id} absent from "
                          f"tracker {self.slot_zone_absent_frames[slot]}f >= "
                          f"{config.DEPARTURE_FRAMES_THRESHOLD}f → departed")
                    self.slot_departed[slot] = True
        # ─────────────────────────────────────────────────────────────────────

        self.active_ids_in_zone = interacting_ids
        self._update_improper_positioning(pose_results)

        # Catch if P1 (id_a) leaves or drops contact and comes back to interact with the door again
        id_a = self.session.get("id_a")
        if id_a is not None and self.session.get("id_b") is None:
            if "id_a_left_zone" not in self.session:
                self.session["id_a_left_zone"] = False
            if "id_a_dropped_contact" not in self.session:
                self.session["id_a_dropped_contact"] = False
                
            if id_a in pose_results:
                res = pose_results[id_a]
                # "At the door" means head or shoulders are in DOOR_ROI
                at_door = res.get("head_in_door", False) or res.get("shoulders_in_door", False)
                
                if not at_door:
                    self.session["id_a_left_zone"] = True
                if not res.get("has_lock_contact"):
                    self.session["id_a_dropped_contact"] = True
                    
                if self.session["id_a_left_zone"] or self.session["id_a_dropped_contact"]:
                    at_door_contact = (
                        (res.get("head_in_door") or res.get("shoulders_in_door"))
                        and res.get("has_lock_contact")
                    )
                    if at_door_contact:
                        timer = self.session.get("same_id_return_timer_frames", 0)
                        timer = min(timer + frame_step, self.min_unlock_frames)
                        self.session["same_id_return_timer_frames"] = timer
                        self.session["same_id_return_grace_frames"] = config.GRACE_BUFFER_FRAMES
                        if timer >= self.min_unlock_frames:
                            print(f"[VIOLATION] P1 (ID {id_a}) confirmed SAME_ID re-attempt after {self.min_unlock_frames / self.fps:.1f}s")
                            self.session["violation_type"] = "SAME_ID"
                    else:
                        grace = self.session.get("same_id_return_grace_frames", 0)
                        timer = self.session.get("same_id_return_timer_frames", 0)
                        if grace > 0 and timer > 0:
                            self.session["same_id_return_grace_frames"] = max(grace - frame_step, 0)
                            # timer pauses — holds value through brief dropout
                        elif timer > 0:
                            # grace expired — P1 disengaged, reset
                            self.session["same_id_return_timer_frames"] = 0
                            self.session["same_id_return_grace_frames"] = 0
                        if res.get("has_lock_contact"):
                            print(f"[DEBUG] P1 (ID {id_a}) lock contact but not at door. head_in_door={res.get('head_in_door')} shoulders_in_door={res.get('shoulders_in_door')}")
            else:
                self.session["id_a_left_zone"] = True

        if self.session["id_a"] is None:
            # Exclude P2 ID and anchor so a dropped P1 slot cannot be filled by the verified P2 person.
            excl_ids_a = {self.session["id_b"]} if self.session["id_b"] is not None else set()
            excl_anchors_a = [self.verified_anchors["b"]] if self.verified_anchors["b"] is not None else []
            self._update_unlock_slot("a", interacting_ids, pose_results,
                                     excluded_ids=excl_ids_a,
                                     excluded_anchors=excl_anchors_a,
                                     tracked_persons=tracked_persons,
                                     frame_step=frame_step)
        elif self.session["id_b"] is None:
            self._update_unlock_slot(
                "b",
                interacting_ids,
                pose_results,
                excluded_ids={self.session["id_a"]},
                tracked_persons=tracked_persons,
                frame_step=frame_step,
            )
        else:
            self.session["sequence_state"] = "READY_FOR_DOOR_OPEN"

    def has_priority_activity(self, tracked_persons: Dict[int, Dict]) -> bool:
        """
        Scanner-tier trigger for priority promotion.

        This is intentionally lighter than authorization: one person standing in
        STANDING_ZONE and facing the door is enough to move this camera stream to
        a full-rate priority lane. The full unlocker rules still run in
        update_timers().
        """
        for track_id, person in tracked_persons.items():
            pose = self._evaluate_unlock_pose(person, track_id, log=False)
            is_standing = pose.get("feet_in_standing", False)
            facing_correct = (
                pose.get("shoulder_order_correct", False)
                or pose.get("left_right_order", False)
            )
            if pose.get("qualified", False) or (is_standing and facing_correct):
                return True
        return False

    def _update_unlock_slot(
        self,
        slot: str,
        interacting_ids: Set[int],
        pose_results: Dict[int, Dict],
        excluded_ids: Optional[Set[int]] = None,
        excluded_anchors: Optional[list] = None,
        tracked_persons: Optional[Dict] = None,
        frame_step: int = 1
    ):
        excluded_ids = excluded_ids or set()
        excluded_anchors = [anchor for anchor in (excluded_anchors or []) if anchor is not None]
        candidate_key = f"candidate_{slot}"
        id_key = f"id_{slot}"
        timer_key = f"timer_{slot}_frames"
        timer_seconds_key = f"timer_{slot}_seconds"
        grace_key = f"grace_buffer_{slot}"

        candidate_id = self.session[candidate_key]
        if candidate_id is None:
            choices = self._candidate_choices(interacting_ids, pose_results, excluded_ids, excluded_anchors)
            if not choices:
                return
            candidate_id = choices[0]
            self.session[candidate_key] = candidate_id
            self.slot_anchors[slot] = pose_results[candidate_id]["anchor"]
            self.session[timer_key] = 0
            self.session[timer_seconds_key] = 0.0
            self.session[grace_key] = config.GRACE_BUFFER_FRAMES
            self.session["sequence_state"] = (
                "FIRST_UNLOCKER_ACTIVE" if slot == "a" else "SECOND_UNLOCKER_ACTIVE"
            )
            # Freeze candidate bbox for height gate in _find_matching_track
            if tracked_persons is not None and candidate_id in tracked_persons:
                self.candidate_bbox[slot] = tracked_persons[candidate_id].get("bbox")
            print(f"[TIMER] P{1 if slot == 'a' else 2} unlock pose confirmed, ID {candidate_id}")

        if candidate_id in excluded_ids:
            self._reset_candidate(slot)
            self.session["violation_type"] = "SAME_ID"
            return

        if candidate_id not in interacting_ids:
            remapped_id = self._find_matching_track(
                self.slot_anchors[slot],
                interacting_ids,
                pose_results,
                excluded_ids,
                excluded_anchors,
                ref_bbox=self.candidate_bbox[slot],
                tracked_persons=tracked_persons,
            )
            if remapped_id is not None:
                print(f"[TRACK] P{1 if slot == 'a' else 2} remapped {candidate_id} → {remapped_id}")
                candidate_id = remapped_id
                self.session[candidate_key] = candidate_id
            else:
                print(f"[TRACK] P{1 if slot == 'a' else 2} ID {candidate_id} lost, no match in interacting_ids={sorted(interacting_ids)}")

        current_timer = self.session[timer_key]
        if candidate_id in interacting_ids:
            self.session[timer_key] = min(current_timer + frame_step, self.max_unlock_frames)
            self.session[timer_seconds_key] = self.session[timer_key] / self.fps
            self.session[grace_key] = config.GRACE_BUFFER_FRAMES
            self.slot_anchors[slot] = self._smooth_anchor(self.slot_anchors[slot], pose_results[candidate_id]["anchor"])

            if self.session[timer_key] >= self.max_unlock_frames:
                self._complete_unlock_slot(slot)
            return

        if current_timer >= self.min_unlock_frames:
            self._complete_unlock_slot(slot)
            return

        if self.session[grace_key] > 0 and current_timer > 0:
            self.session[grace_key] = max(self.session[grace_key] - frame_step, 0)
            print(f"[GRACE] P{1 if slot == 'a' else 2} grace buffer {self.session[grace_key]} frames, timer {current_timer}s")
            return

        if current_timer > 0:
            print(f"[TIMER] P{1 if slot == 'a' else 2} disqualified at {current_timer}s, candidate lost - reset")
        self._reset_candidate(slot)

    def _complete_unlock_slot(self, slot: str):
        candidate_key = f"candidate_{slot}"
        id_key = f"id_{slot}"
        timer_key = f"timer_{slot}_frames"
        timer_seconds_key = f"timer_{slot}_seconds"
        grace_key = f"grace_buffer_{slot}"
        candidate_id = self.session[candidate_key]

        if candidate_id is None:
            return

        self.session[id_key] = candidate_id
        self.assign_unlocker_tag(candidate_id, slot)
        self.verified_anchors[slot] = self.slot_anchors[slot]
        if self.last_seen_bbox[slot] is not None:
            self.slot_height_ref[slot] = self.last_seen_bbox[slot]
        self.session[candidate_key] = None
        self.session[grace_key] = 0
        self.session[timer_key] = min(self.session[timer_key], self.max_unlock_frames)
        self.session[timer_seconds_key] = self.session[timer_key] / self.fps
        self.session["sequence_state"] = (
            "WAITING_FOR_SECOND_UNLOCKER" if slot == "a" else "READY_FOR_DOOR_OPEN"
        )
        print(
            f"[AUTH] P{1 if slot == 'a' else 2} verified "
            f"after {self.session[timer_seconds_key]:.1f}s unlock interaction"
        )

    def _reset_candidate(self, slot: str):
        self.session[f"candidate_{slot}"] = None
        self.session[f"timer_{slot}_frames"] = 0
        self.session[f"timer_{slot}_seconds"] = 0.0
        self.session[f"grace_buffer_{slot}"] = 0
        self.slot_anchors[slot] = None
        self.candidate_bbox[slot] = None

    def _reset_all_timers(self):
        self.session.update(config.create_session())
        self.active_ids_in_zone = set()
        self.slot_anchors = {"a": None, "b": None}
        self.verified_anchors = {"a": None, "b": None}
        self.unlocker_tags = {}
        self.all_unlocker_ids = {"P1_unlocker": set(), "P2_unlocker": set()}
        self.body_fingerprints = {"a": None, "b": None}
        self.slot_lost_frames = {"a": 0, "b": 0}
        self.last_seen_bbox = {"a": None, "b": None}
        self.slot_height_ref = {"a": None, "b": None}
        self.candidate_bbox = {"a": None, "b": None}
        self.slot_departed = {"a": False, "b": False}
        self.slot_zone_absent_frames = {"a": 0, "b": 0}
        # Reset clearance gate — stale standing-zone state must not carry into a new session
        self.standing_zone_was_occupied = False
        self.clearance_frames_since_empty = 0
        self.door_roi_person_overlap = False


    def _refresh_verified_slots(self, pose_results: Dict[int, Dict]):
        for slot in ("a", "b"):
            id_key = f"id_{slot}"
            current_id = self.session.get(id_key)
            anchor = self.verified_anchors[slot]
            if current_id is None or anchor is None:
                continue
            if current_id in pose_results:
                self.verified_anchors[slot] = self._smooth_anchor(anchor, pose_results[current_id]["anchor"])
                continue
            # id_a/id_b are frozen after assignment — no spatial remapping here.
            # Visualization layer (crop-reid) handles display-level re-identification.

    def _candidate_choices(self, interacting_ids, pose_results, excluded_ids, excluded_anchors):
        choices = []
        for track_id in sorted(interacting_ids - excluded_ids):
            anchor = pose_results[track_id]["anchor"]
            if self._matches_any_anchor(anchor, excluded_anchors):
                continue
            choices.append(track_id)
        return choices

    def _find_matching_track(self, anchor, interacting_ids, pose_results, excluded_ids, excluded_anchors,
                              ref_bbox=None, tracked_persons=None):
        if anchor is None:
            return None

        ref_h = float(ref_bbox[3] - ref_bbox[1]) if ref_bbox is not None else None

        best_id = None
        best_distance = float("inf")
        for track_id in sorted(interacting_ids - excluded_ids):
            candidate_anchor = pose_results[track_id]["anchor"]
            if self._matches_any_anchor(candidate_anchor, excluded_anchors):
                continue

            # Height gate: skip if candidate's body height differs >40% from reference.
            # Prevents wrong-person pick in crowded scenes where two people are near the lock.
            if ref_h is not None and tracked_persons is not None:
                cand_bbox = (tracked_persons.get(track_id) or {}).get("bbox")
                if cand_bbox is not None:
                    cand_h = float(cand_bbox[3] - cand_bbox[1])
                    if ref_h > 0 and abs(cand_h - ref_h) / ref_h > 0.4:
                        continue

            distance = self._anchor_distance(anchor, candidate_anchor)
            if distance < best_distance:
                best_id = track_id
                best_distance = distance

        if best_id is not None and best_distance <= config.UNLOCKER_ANCHOR_MATCH_PIXELS:
            return best_id
        return None

    def _matches_any_anchor(self, anchor, anchors) -> bool:
        return any(self._anchor_distance(anchor, other) <= config.UNLOCKER_ANCHOR_MATCH_PIXELS for other in anchors)

    def _anchor_distance(self, a, b) -> float:
        if a is None or b is None:
            return float("inf")
        return math.dist(a, b)

    def _smooth_anchor(self, previous, current):
        if previous is None:
            return current
        return (
            previous[0] * 0.65 + current[0] * 0.35,
            previous[1] * 0.65 + current[1] * 0.35,
        )

    # ================================================================
    # UNLOCK POSE RULES
    # ================================================================
    def _evaluate_unlock_pose(self, person: Dict, track_id: int = None, log: bool = True) -> Dict:
        """
        Evaluate whether this person's pose qualifies them for an unlock slot.

        Sequential dual-lock gate (A → B):
          Step 1 — Both wrists (wrist-first; elbow as fallback) must be near LOCK_A
                   simultaneously. Once confirmed, a_touched is latched True.
          Step 2 — After a_touched, both wrists must be near LOCK_B simultaneously.
                   Once confirmed, b_touched is latched True and the person qualifies.

        After completing A→B, the person stays qualified as long as their head is
        in INTERACTION_ZONE and arms are raised (grace buffer handles brief occlusion).
        """
        keypoints = person.get("keypoints")
        bbox = person.get("bbox")

        result = {
            "qualified": False,
            "head_in_interaction": False,
            "feet_in_standing": False,
            "waist_near_door": False,      # OPTIONAL — hip inside DOOR_ROI
            "ear_order_correct": False,    # MANDATORY when both ears visible
            "shoulder_order_correct": False, # MANDATORY when shoulders used for door gate
            "in_locks_roi": False,
            "arms_raised": False,
            "left_right_order": False,
            "has_lock_contact": False,
            "head_in_door": False,
            "shoulders_in_door": False,
            "anchor": None,
        }

        if keypoints is None or bbox is None or len(keypoints) <= config.KEYPOINT_ANKLE_RIGHT:
            return result

        head_pos = self._get_head_position(keypoints)
        if head_pos is not None:
            result["head_in_interaction"] = self.roi_manager.point_in_roi(
                "INTERACTION_ZONE", head_pos[0], head_pos[1]
            )
            result["head_in_door"] = self.roi_manager.point_in_roi(
                "DOOR_ROI", head_pos[0], head_pos[1]
            )

        fx, fy = self._get_base_position(keypoints, bbox)
        result["feet_in_standing"] = self.roi_manager.point_in_roi("STANDING_ZONE", fx, fy)
        result["anchor"] = self._person_anchor(keypoints, bbox, head_pos, (fx, fy))
        result["waist_near_door"] = self._waist_near_door(keypoints, bbox)  # OPTIONAL
        result["arms_raised"] = self._arms_raised_towards_door(keypoints, bbox)
        result["left_right_order"] = self._left_right_keypoints_in_video_order(keypoints)
        result["shoulders_in_door"] = self._shoulders_in_door(keypoints)
        result["shoulder_order_correct"] = self._shoulder_order_correct(keypoints)
        result["ear_order_correct"] = self._ear_order_correct(keypoints)

        # All 4 keypoints must be in LOCKS_ROI simultaneously
        result["in_locks_roi"] = self._all_arm_keypoints_in_locks_roi(keypoints)
        result["has_lock_contact"] = result["in_locks_roi"]

        # ── Qualification gate ─────────────────────────────────────────────────
        # MANDATORY : in_locks_roi, (head_in_door OR (shoulders_in_door AND shoulder_order_correct)),
        #             ear_order_correct, arms_raised
        # OPTIONAL  : feet_in_standing, left_right_order, waist_near_door
        if result["in_locks_roi"]:
            shoulders_ok = result["shoulders_in_door"] and result["shoulder_order_correct"]
            result["qualified"] = (
                (result["head_in_door"] or shoulders_ok)
                and result["ear_order_correct"]
                and result["arms_raised"]
            )
        else:
            result["qualified"] = False

        # ── Debug logging ──────────────────────────────────────────────────────
        if log:
            id_tag = f" ID {track_id}" if track_id is not None else ""
            if result["qualified"]:
                optional_notes = []
                if not result["feet_in_standing"]:
                    optional_notes.append("feet_not_in_standing")
                if not result["left_right_order"]:
                    optional_notes.append("elbow_order_skipped")
                if not result["waist_near_door"]:
                    optional_notes.append("waist_not_near_door")
                note = f" (optional skipped: {', '.join(optional_notes)})" if optional_notes else ""
                print(f"[POSE]{id_tag} Qualified: ears✓ shoulders✓ in_locks✓ head✓ arms✓{note}")
            else:
                failed = []
                if not result["ear_order_correct"]:
                    failed.append("ear_order_wrong")
                if not result["shoulder_order_correct"] and result["shoulders_in_door"]:
                    failed.append("shoulder_order_wrong")
                if not (result["head_in_door"] or (result["shoulders_in_door"] and result["shoulder_order_correct"])):
                    failed.append("not_in_door_roi")
                if not result["in_locks_roi"]:
                    failed.append("not_in_locks_roi")
                if not result["arms_raised"]:
                    failed.append("arms_not_raised")
                optional = []
                if not result["feet_in_standing"]:
                    optional.append("feet_not_in_standing")
                if not result["left_right_order"]:
                    optional.append("elbow_order_wrong")
                if not result["waist_near_door"]:
                    optional.append("waist_not_near_door")
                opt_str = f" | Optional: {', '.join(optional)}" if optional else ""
                print(f"[POSE]{id_tag} waiting: {', '.join(failed)}{opt_str}")

        return result

    def _ear_order_correct(self, keypoints: np.ndarray) -> bool:
        """Verify left/right ear orientation matches expected camera perspective.

        Right ear must appear to the RIGHT, left ear to the LEFT in the video frame
        (reversed for mirror_left_right cameras). This confirms the person is correctly
        oriented towards the door before any lock interaction progress is credited.

        Graceful fallback:
          - Both ears invisible → True  (cannot determine; don't block)
          - Only one ear visible → True (ambiguous; don't block)
          - Both visible → enforce order with LEFT_RIGHT_ORDER_MIN_PIXELS margin
        """
        left_ear  = self._visible_keypoint(keypoints, config.KEYPOINT_EAR_LEFT)
        right_ear = self._visible_keypoint(keypoints, config.KEYPOINT_EAR_RIGHT)

        if left_ear is None or right_ear is None:
            # Cannot determine orientation — don't block the person
            return True

        if self.mirror_left_right:
            # Top-down / mirrored camera: body-left ear appears on video RIGHT
            return left_ear[0] > right_ear[0] + config.LEFT_RIGHT_ORDER_MIN_PIXELS
        else:
            # Standard camera: body-left ear appears on video LEFT
            return left_ear[0] < right_ear[0] - config.LEFT_RIGHT_ORDER_MIN_PIXELS

    def _waist_near_door(self, keypoints: np.ndarray, bbox) -> bool:
        """OPTIONAL check — hip keypoint inside DOOR_ROI.
        Top-down camera angles frequently place hip keypoints outside the door polygon
        even for correctly positioned persons; this check must never block qualification.
        """
        hr_x, hr_y, hr_c = keypoints[config.KEYPOINT_HIP_RIGHT]
        hl_x, hl_y, hl_c = keypoints[config.KEYPOINT_HIP_LEFT]

        waist_pos = None
        if hr_c >= config.ARM_KEYPOINT_CONFIDENCE_THRESHOLD:
            waist_pos = (float(hr_x), float(hr_y))
        elif hl_c >= config.ARM_KEYPOINT_CONFIDENCE_THRESHOLD:
            waist_pos = (float(hl_x), float(hl_y))

        if waist_pos is None:
            return False

        return self.roi_manager.point_in_roi("DOOR_ROI", waist_pos[0], waist_pos[1])

    def _update_improper_positioning(self, pose_results: Dict[int, Dict]):
        self.session["improper_positioning"] = None
        for track_id, pose in pose_results.items():
            # feet_in_standing is optional, so only head_in_interaction is required for proper positioning
            if pose["has_lock_contact"] and not pose["head_in_interaction"]:
                self.session["improper_positioning"] = track_id
                return

    def _get_head_position(self, keypoints: np.ndarray) -> Optional[Tuple[float, float]]:
        points = []
        for idx in HEAD_KEYPOINTS:
            x, y, conf = keypoints[idx]
            if conf >= config.HEAD_CONFIDENCE_THRESHOLD:
                points.append((float(x), float(y)))
        if not points:
            return None
        arr = np.array(points, dtype=float)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())

    def _all_arm_keypoints_in_locks_roi(self, keypoints: np.ndarray) -> bool:
        """Check if all 4 arm keypoints (LW, LE, RW, RL) are visible and in LOCKS_ROI."""
        indices = [
            config.KEYPOINT_WRIST_LEFT,
            config.KEYPOINT_ELBOW_LEFT,
            config.KEYPOINT_WRIST_RIGHT,
            config.KEYPOINT_ELBOW_RIGHT
        ]
        
        pts_in = 0
        for idx in indices:
            pt = self._visible_keypoint(keypoints, idx)
            if pt is not None and self.roi_manager.point_in_roi("LOCKS_ROI", pt[0], pt[1]):
                pts_in += 1
        
        if pts_in > 0 and pts_in < 4:
            # Only print every 15 frames to avoid spam
            if getattr(self, "_arm_pts_tick", 0) % 15 == 0:
                print(f"[POSE] Interaction weak: {pts_in}/4 arm points in LOCKS_ROI")
            self._arm_pts_tick = getattr(self, "_arm_pts_tick", 0) + 1
            
        return pts_in >= 4

    def _left_right_keypoints_in_video_order(self, keypoints: np.ndarray) -> bool:
        """
        Door-facing/back-to-camera check using elbow keypoints primarily.
        Ankle check is included only when both ankles are visible (confidence >= threshold).
        - Standard (mirror_left_right=False): body-left elbow should appear LEFT in video frame.
        - Mirrored (mirror_left_right=True): for top-down cameras where the person faces away,
          body-left keypoints appear on the RIGHT side of the video frame.
        Note: This check is now OPTIONAL in the qualification gate — low-confidence ankle
        keypoints in top-down cameras would otherwise block all valid unlockers.
        """
        # Elbow check is the primary check — always required for this function to return True
        left_elbow = self._visible_keypoint(keypoints, config.KEYPOINT_ELBOW_LEFT)
        right_elbow = self._visible_keypoint(keypoints, config.KEYPOINT_ELBOW_RIGHT)

        if left_elbow is None or right_elbow is None:
            return False

        if self.mirror_left_right:
            if left_elbow[0] <= right_elbow[0] + config.LEFT_RIGHT_ORDER_MIN_PIXELS:
                return False
        else:
            if right_elbow[0] <= left_elbow[0] + config.LEFT_RIGHT_ORDER_MIN_PIXELS:
                return False

        # Ankle check: only enforce if BOTH ankles are visible
        left_ankle = self._visible_keypoint(keypoints, config.KEYPOINT_ANKLE_LEFT)
        right_ankle = self._visible_keypoint(keypoints, config.KEYPOINT_ANKLE_RIGHT)

        if left_ankle is not None and right_ankle is not None:
            if self.mirror_left_right:
                if left_ankle[0] <= right_ankle[0] + config.LEFT_RIGHT_ORDER_MIN_PIXELS:
                    return False
            else:
                if right_ankle[0] <= left_ankle[0] + config.LEFT_RIGHT_ORDER_MIN_PIXELS:
                    return False

        return True

    def _visible_keypoint(self, keypoints: np.ndarray, idx: int) -> Optional[Tuple[float, float]]:
        if idx >= len(keypoints):
            return None
        x, y, conf = keypoints[idx]
        if conf < config.ARM_KEYPOINT_CONFIDENCE_THRESHOLD:
            return None
        return float(x), float(y)

    def _shoulders_in_door(self, keypoints: np.ndarray) -> bool:
        """Position only: True if either shoulder keypoint is inside DOOR_ROI."""
        left_shoulder  = self._visible_keypoint(keypoints, config.KEYPOINT_SHOULDER_LEFT)
        right_shoulder = self._visible_keypoint(keypoints, config.KEYPOINT_SHOULDER_RIGHT)
        if left_shoulder and self.roi_manager.point_in_roi("DOOR_ROI", left_shoulder[0], left_shoulder[1]):
            return True
        if right_shoulder and self.roi_manager.point_in_roi("DOOR_ROI", right_shoulder[0], right_shoulder[1]):
            return True
        return False

    def _shoulder_order_correct(self, keypoints: np.ndarray) -> bool:
        """Orientation only: right shoulder RIGHT, left shoulder LEFT in video frame.

        Both visible  → enforce relative order with LEFT_RIGHT_ORDER_MIN_PIXELS margin.
        One visible   → enforce absolute side using head center as person midline.
        Neither       → True (graceful fallback, can't determine).
        """
        left_shoulder  = self._visible_keypoint(keypoints, config.KEYPOINT_SHOULDER_LEFT)
        right_shoulder = self._visible_keypoint(keypoints, config.KEYPOINT_SHOULDER_RIGHT)

        if left_shoulder is None and right_shoulder is None:
            return True  # no shoulders visible → don't block

        # Both visible → relative order
        if left_shoulder is not None and right_shoulder is not None:
            if self.mirror_left_right:
                return left_shoulder[0] > right_shoulder[0] + config.LEFT_RIGHT_ORDER_MIN_PIXELS
            else:
                return left_shoulder[0] < right_shoulder[0] - config.LEFT_RIGHT_ORDER_MIN_PIXELS

        # Single shoulder → absolute side using head midline
        head_pos = self._get_head_position(keypoints)
        if head_pos is None:
            return True  # no reference → don't block

        mid_x = head_pos[0]
        if right_shoulder is not None:
            return right_shoulder[0] > mid_x if not self.mirror_left_right else right_shoulder[0] < mid_x
        else:
            return left_shoulder[0] < mid_x if not self.mirror_left_right else left_shoulder[0] > mid_x

    def _point_in_or_near_roi(self, roi_name: str, point: Tuple[float, float], threshold: float) -> bool:
        x, y = point
        if self.roi_manager.point_in_roi(roi_name, x, y):
            return True

        center = self.roi_manager.get_roi_center(roi_name)
        if center is None:
            return False
        return math.dist(point, center) <= threshold

    def _arms_raised_towards_door(self, keypoints: np.ndarray, bbox) -> bool:
        left_y = self._highest_visible_y(
            keypoints, (config.KEYPOINT_WRIST_LEFT, config.KEYPOINT_ELBOW_LEFT)
        )
        right_y = self._highest_visible_y(
            keypoints, (config.KEYPOINT_WRIST_RIGHT, config.KEYPOINT_ELBOW_RIGHT)
        )
        hip_y = self._body_reference_y(keypoints, bbox)

        if left_y is None or right_y is None or hip_y is None:
            return False

        return (
            left_y < hip_y - config.DOOR_FACING_ARM_RAISE_PIXELS
            and right_y < hip_y - config.DOOR_FACING_ARM_RAISE_PIXELS
        )

    def _highest_visible_y(self, keypoints: np.ndarray, indices) -> Optional[float]:
        ys = []
        for idx in indices:
            point = self._visible_keypoint(keypoints, idx)
            if point is not None:
                ys.append(point[1])
        return min(ys) if ys else None

    def _body_reference_y(self, keypoints: np.ndarray, bbox) -> Optional[float]:
        hip_ys = []
        for idx in (config.KEYPOINT_HIP_LEFT, config.KEYPOINT_HIP_RIGHT):
            if idx < len(keypoints):
                _, y, conf = keypoints[idx]
                if conf >= config.ARM_KEYPOINT_CONFIDENCE_THRESHOLD:
                    hip_ys.append(float(y))
        if hip_ys:
            return float(np.mean(hip_ys))
        return float(bbox[3]) if bbox is not None else None

    def _person_anchor(self, keypoints: np.ndarray, bbox, head_pos, base_pos) -> Tuple[float, float]:
        anchor_points = []
        if head_pos is not None:
            anchor_points.append(head_pos)
        if base_pos is not None:
            anchor_points.append(base_pos)

        for idx in (
            config.KEYPOINT_ELBOW_LEFT,
            config.KEYPOINT_ELBOW_RIGHT,
            config.KEYPOINT_ANKLE_LEFT,
            config.KEYPOINT_ANKLE_RIGHT,
        ):
            point = self._visible_keypoint(keypoints, idx)
            if point is not None:
                anchor_points.append(point)

        if not anchor_points:
            return self._bbox_bottom_center(bbox)
        arr = np.array(anchor_points, dtype=float)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())

    # ================================================================
    # GEOMETRY HELPERS
    # ================================================================
    def _get_base_position(self, keypoints: np.ndarray, bbox) -> Tuple[float, float]:
        ar_x, ar_y, ar_c = keypoints[config.KEYPOINT_ANKLE_RIGHT]
        al_x, al_y, al_c = keypoints[config.KEYPOINT_ANKLE_LEFT]
        if ar_c >= config.ANKLE_CONFIDENCE_THRESHOLD:
            return float(ar_x), float(ar_y)
        if al_c >= config.ANKLE_CONFIDENCE_THRESHOLD:
            return float(al_x), float(al_y)

        hr_x, hr_y, hr_c = keypoints[config.KEYPOINT_HIP_RIGHT]
        hl_x, hl_y, hl_c = keypoints[config.KEYPOINT_HIP_LEFT]
        if hr_c >= config.HIP_FALLBACK_THRESHOLD:
            return float(hr_x), float(hr_y)
        if hl_c >= config.HIP_FALLBACK_THRESHOLD:
            return float(hl_x), float(hl_y)

        return self._bbox_bottom_center(bbox)

    def _bbox_bottom_center(self, bbox) -> Tuple[float, float]:
        return float((bbox[0] + bbox[2]) / 2), float(bbox[3])

    def _bbox_overlaps_door_roi(self, bbox) -> bool:
        x1, y1, x2, y2 = bbox
        points_to_check = [
            ((x1 + x2) / 2, (y1 + y2) / 2),
            ((x1 + x2) / 2, y2),
            (x1, y2),
            (x2, y2),
        ]
        return any(self.roi_manager.point_in_roi("DOOR_ROI", px, py) for px, py in points_to_check)

    # ================================================================
    # OCCLUSION FALLBACK
    # ================================================================
    def _min_distance_to_polygon(self, roi_name: str, point: Tuple[float, float]) -> float:
        roi = self.roi_manager.get_roi(roi_name)
        if roi is None:
            return float("inf")

        min_dist = float("inf")
        for i in range(len(roi)):
            p1 = roi[i]
            p2 = roi[(i + 1) % len(roi)]
            dist = self._point_to_segment_distance(point, p1, p2)
            min_dist = min(min_dist, dist)
        return min_dist

    def _point_to_segment_distance(self, point: Tuple[float, float], seg_start: Tuple[float, float], seg_end: Tuple[float, float]) -> float:
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end

        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.dist(point, seg_start)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        return math.dist(point, (closest_x, closest_y))

    def _check_occlusion_fallback(self, person: Dict, track_id: int) -> bool:
        keypoints = person.get("keypoints")
        if keypoints is None or len(keypoints) <= config.KEYPOINT_ANKLE_RIGHT:
            return False

        ar_conf = keypoints[config.KEYPOINT_ANKLE_RIGHT][2]
        al_conf = keypoints[config.KEYPOINT_ANKLE_LEFT][2]
        if not (ar_conf < config.ANKLE_OCCLUSION_CONFIDENCE_THRESHOLD and
                al_conf < config.ANKLE_OCCLUSION_CONFIDENCE_THRESHOLD):
            return False

        if not self._left_right_keypoints_in_video_order(keypoints):
            return False

        head_pos = self._get_head_position(keypoints)
        if head_pos is None:
            return False

        min_dist = self._min_distance_to_polygon("LOCKS_ROI", head_pos)
        if min_dist > config.HEAD_TO_LOCKS_MAX_PIXELS:
            return False

        if not self._all_arm_keypoints_in_locks_roi(keypoints):
            return False

        return True

    def _apply_occlusion_recovery(self, tracked_persons: Dict, pose_results: Dict):
        for slot in ("a", "b"):
            id_key = f"id_{slot}"
            assigned_id = self.session.get(id_key)
            if assigned_id is None:
                continue

            if assigned_id not in pose_results:
                continue

            pose = pose_results[assigned_id]
            if pose["qualified"]:
                continue

            person = tracked_persons.get(assigned_id)
            if person is None:
                continue

            if self._check_occlusion_fallback(person, assigned_id):
                pose_results[assigned_id]["qualified"] = True
                pose_results[assigned_id]["occlusion_mode"] = True
                print(f"[OCCLUSION] P{1 if slot == 'a' else 2} fallback qualified")

    # ================================================================
    # VERIFIED UNLOCKER ReID (state-machine level)
    # ================================================================
    def _remap_verified_unlocker(self, slot: str, tracked_persons: Dict, pose_results: Dict) -> bool:
        """
        When a verified unlocker's ID is lost, find the best spatial match in
        tracked_persons and re-assign session["id_slot"] to that ID.

        Matching criteria (strict):
          1. Anchor distance ≤ UNLOCKER_ANCHOR_MATCH_PIXELS × 3 (285px)
          2. Bbox height within 40% of last seen bbox
          3. Not the other verified unlocker's ID or tagged with other slot

        Returns True if re-assignment was made.
        """
        if self.slot_departed[slot]:
            return False  # departed unlocker must not be silently remapped to a stranger

        id_key = f"id_{slot}"
        current_id = self.session.get(id_key)
        anchor = self.verified_anchors[slot]
        ref_bbox = self.last_seen_bbox[slot]
        # Use slot_height_ref (frozen at verification) for size gate so a partial
        # edge-clip bbox on last_seen_bbox doesn't wrongly reject a full-body re-entry.
        height_ref_bbox = self.slot_height_ref[slot] or ref_bbox

        anchor = self.verified_anchors[slot]
        if anchor is None:
            return False

        other_slot = "b" if slot == "a" else "a"
        other_id = self.session.get(f"id_{other_slot}")
        other_tag = f"P{2 if slot == 'a' else 1}_unlocker"

        # Use last_seen_bbox center-bottom as live search origin.
        # verified_anchor is frozen at the lock interaction position — after P2 moves
        # into the room it becomes useless. last_seen_bbox tracks the most recent
        # confirmed position so the search follows the person's movement.
        if ref_bbox is not None:
            search_origin = ((ref_bbox[0] + ref_bbox[2]) / 2.0, float(ref_bbox[3]))
        else:
            search_origin = anchor  # fallback when no bbox history yet

        # Radius grows 12px per lost frame so brief occlusion stays tight while
        # a slow-moving person crossing the doorway gets found after a few frames.
        frames_lost = self.slot_lost_frames.get(slot, 0)
        search_radius = min(config.UNLOCKER_ANCHOR_MATCH_PIXELS * 3 + frames_lost * 12, 600)

        best_id = None
        best_dist = search_radius
        best_bbox = None
        best_anchor = None

        for tid, person in tracked_persons.items():
            if tid == current_id:
                continue
            if tid == other_id:
                continue
            if self.unlocker_tags.get(tid) == other_tag:
                continue

            bbox = person.get("bbox")
            if bbox is None:
                continue

            # Bbox size gate — use height_ref_bbox (frozen at verification) so partial
            # edge-clip frames don't shrink the reference and reject a valid full-body return.
            if height_ref_bbox is not None:
                ref_h = float(height_ref_bbox[3] - height_ref_bbox[1])
                cand_h = float(bbox[3] - bbox[1])
                if ref_h > 0 and abs(cand_h - ref_h) / ref_h > 0.4:
                    continue

            keypoints = person.get("keypoints")
            head_pos = self._get_head_position(keypoints) if keypoints is not None else None
            base_pos = self._get_base_position(keypoints, bbox) if keypoints is not None else self._bbox_bottom_center(bbox)
            cand_anchor = self._person_anchor(keypoints, bbox, head_pos, base_pos) if keypoints is not None else self._bbox_bottom_center(bbox)

            dist = math.dist(search_origin, cand_anchor)
            if dist < best_dist:
                best_dist = dist
                best_id = tid
                best_bbox = bbox
                best_anchor = cand_anchor

        if best_id is None:
            return False

        old_id = current_id
        self.session[id_key] = best_id
        self.assign_unlocker_tag(best_id, slot)
        if best_bbox is not None:
            self.last_seen_bbox[slot] = best_bbox
        if best_anchor is not None:
            self.verified_anchors[slot] = self._smooth_anchor(anchor, best_anchor)

        print(f"[REID-SM] P{1 if slot == 'a' else 2} verified unlocker: ID {old_id} → {best_id} "
              f"(dist={best_dist:.1f}px)")
        return True

    # ================================================================
    # UNLOCKER TAGGING (Redundant ID tracking per person)
    # ================================================================
    def assign_unlocker_tag(self, track_id: int, slot: str) -> None:
        """
        Tag a track ID with unlocker slot. Multiple IDs can share same tag.
        This enables ID continuity: if track_id is lost, a new ID can get same tag.

        Args:
            track_id: ByteTrack ID to tag
            slot: 'a' (P1) or 'b' (P2)
        """
        tag = f"P{1 if slot == 'a' else 2}_unlocker"
        self.unlocker_tags[track_id] = tag
        self.all_unlocker_ids[tag].add(track_id)
        print(f"[TAG] ID {track_id} tagged as {tag}")

    def get_all_ids_for_tag(self, tag: str) -> Set[int]:
        """Get all IDs ever assigned to this unlocker."""
        return self.all_unlocker_ids.get(tag, set()).copy()


    # ================================================================
    # AUTHORIZATION
    # ================================================================
    def check_authorization(self) -> Dict:
        id_a = self.session["id_a"]
        id_b = self.session["id_b"]
        lock_a_auth = id_a is not None and self.session["timer_a_frames"] >= self.min_unlock_frames
        lock_b_auth = id_b is not None and self.session["timer_b_frames"] >= self.min_unlock_frames

        authorized = lock_a_auth and lock_b_auth and id_a != id_b

        violation = self.session.get("violation_type")
        if not authorized and violation is None:
            violation = "INCOMPLETE"

        return {
            "authorized": authorized,
            "lock_a_authorized": lock_a_auth,
            "lock_b_authorized": lock_b_auth,
            "violation_type": None if authorized else violation,
        }

    def verified_unlockers_in_interaction_zone(self, tracked_persons: Dict[int, Dict]) -> bool:
        """Return True when verified P1 and P2 are both in INTERACTION_ZONE.
        Falls back to last_seen_head_pos when tracker drops an ID (e.g. occlusion at doorway).
        Returns False immediately if either unlocker has been marked departed — a different
        physical person inheriting the same tracker ID does not satisfy this check."""
        for slot in ("a", "b"):
            track_id = self.session.get(f"id_{slot}")
            if track_id is None:
                return False

            if self.slot_departed[slot]:
                print(f"[ZONE] P{1 if slot == 'a' else 2} marked departed → both_in_interaction_zone=False")
                return False

            head_pos = None
            if track_id in tracked_persons:
                keypoints = tracked_persons[track_id].get("keypoints")
                if keypoints is not None:
                    head_pos = self._get_head_position(keypoints)

            if head_pos is None:
                # Fallback to last seen head position ONLY if lost very recently (e.g. brief occlusion)
                if self.slot_lost_frames.get(slot, 0) <= config.MAX_SYNTHETIC_HOLD_FRAMES:
                    head_pos = self.last_seen_head_pos.get(slot)

            if head_pos is None:
                return False

            if not self.roi_manager.point_in_roi("INTERACTION_ZONE", head_pos[0], head_pos[1]):
                return False

        return True

    def reset_session(self):
        self._reset_all_timers()
        self.current_frame_count += 1
