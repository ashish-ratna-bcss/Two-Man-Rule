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

    def __init__(self, roi_manager: ROIManager, fps: int = 30, session_id: Optional[str] = None):
        self.roi_manager = roi_manager
        self.fps = max(int(fps or config.DEFAULT_FPS), 1)
        self.dwell_frames = config.calculate_min_unlock_frames(self.fps)
        self.min_unlock_frames = self.dwell_frames
        self.max_unlock_frames = config.calculate_max_unlock_frames(self.fps)
        self.session = config.create_session()

        self.active_ids_in_zone: Set[int] = set()
        self.current_frame_count = 0

        # Clearance gate state for door verification.
        self.standing_zone_was_occupied = False
        self.clearance_frames_since_empty = 0
        self.clearance_buffer_frames = 3
        self.door_roi_person_overlap = False

        # Saved once both different unlockers complete their 6-10s interaction.
        self.authorized_session_buffer: Optional[Dict] = None
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
        # Used to validate anchor-based re-ID candidates (same body size = same person)
        self.last_seen_bbox = {"a": None, "b": None}

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

        for track_id, person in tracked_persons.items():
            pose = self._evaluate_unlock_pose(person)
            pose_results[track_id] = pose
            if pose["qualified"]:
                interacting_ids.add(track_id)

        self._apply_occlusion_recovery(tracked_persons, pose_results)

        for track_id, pose in pose_results.items():
            if pose["qualified"]:
                interacting_ids.add(track_id)

        # Once assigned, keep person alive - add to interacting_ids if available, else use grace buffer
        for slot in ("a", "b"):
            assigned_id = self.session.get(f"id_{slot}")
            candidate_id = self.session.get(f"candidate_{slot}")
            timer = self.session.get(f"timer_{slot}_frames", 0)

            # Assigned person: id_a/id_b are FIXED after assignment — never changed here.
            # Visualization layer handles display remapping via crop-reid independently.
            if assigned_id is not None:
                if assigned_id in pose_results:
                    interacting_ids.add(assigned_id)
                    # Refresh last-known bbox while person is directly visible
                    if assigned_id in tracked_persons:
                        bbox = tracked_persons[assigned_id].get("bbox")
                        if bbox is not None:
                            self.last_seen_bbox[slot] = bbox
                else:
                    # Person lost from tracker — hold via synthetic entry (grace buffer handles expiry)
                    print(f"[ASSIGN] P{1 if slot == 'a' else 2} ID {assigned_id} not in tracker, synthetic hold")
                    pose_results[assigned_id] = {
                        "qualified": True,
                        "has_lock_contact": True,
                        "head_in_interaction": True,
                        "feet_in_standing": True,
                        "anchor": self.verified_anchors[slot]
                    }
                    interacting_ids.add(assigned_id)
                continue

            # Candidate: stay qualified if timer started and still tracked
            if candidate_id is not None and timer > 0 and candidate_id in tracked_persons:
                interacting_ids.add(candidate_id)
                print(f"[CANDI] P{1 if slot == 'a' else 2} candidate ID {candidate_id} (timer={timer}f)")

        self._refresh_verified_slots(pose_results)
        self.active_ids_in_zone = interacting_ids
        self._update_improper_positioning(pose_results)

        if self.session["id_a"] is None:
            self._update_unlock_slot("a", interacting_ids, pose_results, frame_step=frame_step)
        elif self.session["id_b"] is None:
            self._update_unlock_slot(
                "b",
                interacting_ids,
                pose_results,
                excluded_ids={self.session["id_a"]},
                frame_step=frame_step,
            )
        else:
            self._validate_both_ids_in_interaction_zone(tracked_persons, pose_results)
            self.session["sequence_state"] = "READY_FOR_DOOR_OPEN"

    def _update_unlock_slot(
        self,
        slot: str,
        interacting_ids: Set[int],
        pose_results: Dict[int, Dict],
        excluded_ids: Optional[Set[int]] = None,
        excluded_anchors: Optional[list] = None,
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

    def _reset_all_timers(self):
        self.session.update(config.create_session())
        self.active_ids_in_zone = set()
        self.slot_anchors = {"a": None, "b": None}
        self.verified_anchors = {"a": None, "b": None}
        self.unlocker_tags = {}
        self.all_unlocker_ids = {"P1_unlocker": set(), "P2_unlocker": set()}
        self.body_fingerprints = {"a": None, "b": None}
        self.last_seen_bbox = {"a": None, "b": None}

    def _validate_both_ids_in_interaction_zone(self, tracked_persons: Dict[int, Dict], pose_results: Dict[int, Dict]):
        """Verify both assigned persons have heads in INTERACTION_ZONE."""
        id_a = self.session.get("id_a")
        id_b = self.session.get("id_b")

        if id_a is not None and id_a in pose_results:
            pose_a = pose_results[id_a]
            if pose_a.get("head_in_interaction"):
                print(f"[VALIDATION] P1 (ID {id_a}) head confirmed in INTERACTION_ZONE")
            else:
                print(f"[VALIDATION] P1 (ID {id_a}) head NOT in INTERACTION_ZONE")

        if id_b is not None and id_b in pose_results:
            pose_b = pose_results[id_b]
            if pose_b.get("head_in_interaction"):
                print(f"[VALIDATION] P2 (ID {id_b}) head confirmed in INTERACTION_ZONE")
            else:
                print(f"[VALIDATION] P2 (ID {id_b}) head NOT in INTERACTION_ZONE")

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

    def _find_matching_track(self, anchor, interacting_ids, pose_results, excluded_ids, excluded_anchors):
        if anchor is None:
            return None

        best_id = None
        best_distance = float("inf")
        for track_id in sorted(interacting_ids - excluded_ids):
            candidate_anchor = pose_results[track_id]["anchor"]
            if self._matches_any_anchor(candidate_anchor, excluded_anchors):
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
    def _evaluate_unlock_pose(self, person: Dict) -> Dict:
        keypoints = person.get("keypoints")
        bbox = person.get("bbox")

        result = {
            "qualified": False,
            "head_in_interaction": False,
            "feet_in_standing": False,
            "waist_near_door": False,
            "contact_a": False,
            "contact_b": False,
            "left_arm_engaged": False,
            "right_arm_engaged": False,
            "arms_raised": False,
            "left_right_order": False,
            "has_lock_contact": False,
            "anchor": None,
        }

        if keypoints is None or bbox is None or len(keypoints) <= config.KEYPOINT_ANKLE_RIGHT:
            return result

        head_pos = self._get_head_position(keypoints)
        if head_pos is not None:
            result["head_in_interaction"] = self.roi_manager.point_in_roi(
                "INTERACTION_ZONE", head_pos[0], head_pos[1]
            )

        fx, fy = self._get_base_position(keypoints, bbox)
        result["feet_in_standing"] = self.roi_manager.point_in_roi("STANDING_ZONE", fx, fy)
        result["anchor"] = self._person_anchor(keypoints, bbox, head_pos, (fx, fy))

        result["waist_near_door"] = self._waist_near_door(keypoints, bbox)

        result["contact_a"] = self._side_contacts_lock(keypoints, "left", "LOCK_A_ROI") or self._side_contacts_lock(
            keypoints, "right", "LOCK_A_ROI"
        )
        result["contact_b"] = self._side_contacts_lock(keypoints, "left", "LOCK_B_ROI") or self._side_contacts_lock(
            keypoints, "right", "LOCK_B_ROI"
        )
        result["left_arm_engaged"] = self._side_contacts_any_lock(keypoints, "left")
        result["right_arm_engaged"] = self._side_contacts_any_lock(keypoints, "right")
        result["arms_raised"] = self._arms_raised_towards_door(keypoints, bbox)
        result["left_right_order"] = self._left_right_keypoints_in_video_order(keypoints)
        result["has_lock_contact"] = result["contact_a"] or result["contact_b"]

        result["qualified"] = (
            result["head_in_interaction"]
            and result["feet_in_standing"]
            and result["waist_near_door"]
            and result["contact_a"]
            and result["contact_b"]
            and result["left_arm_engaged"]
            and result["right_arm_engaged"]
            and result["arms_raised"]
            and result["left_right_order"]
        )

        if result["qualified"]:
            print("[POSE] Qualified: all conditions pass")
        else:
            failed = []
            if not result["head_in_interaction"]:
                failed.append("head_not_in_interaction")
            if not result["feet_in_standing"]:
                failed.append("feet_not_in_standing")
            if not result["waist_near_door"]:
                failed.append("waist_not_near_door")
            if not result["contact_a"]:
                failed.append("no_contact_a")
            if not result["contact_b"]:
                failed.append("no_contact_b")
            if not result["left_arm_engaged"]:
                failed.append("left_arm_not_engaged")
            if not result["right_arm_engaged"]:
                failed.append("right_arm_not_engaged")
            if not result["arms_raised"]:
                failed.append("arms_not_raised")
            if not result["left_right_order"]:
                failed.append("left_right_order_wrong")
            print(f"[POSE] Disqualified: {', '.join(failed)}")

        return result

    def _waist_near_door(self, keypoints: np.ndarray, bbox) -> bool:
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
            if pose["has_lock_contact"] and not (pose["head_in_interaction"] and pose["feet_in_standing"]):
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

    def _side_contacts_any_lock(self, keypoints: np.ndarray, side: str) -> bool:
        return (
            self._side_contacts_lock(keypoints, side, "LOCK_A_ROI")
            or self._side_contacts_lock(keypoints, side, "LOCK_B_ROI")
        )

    def _side_contacts_lock(self, keypoints: np.ndarray, side: str, lock_roi_name: str) -> bool:
        if side == "left":
            wrist_idx = config.KEYPOINT_WRIST_LEFT
            elbow_idx = config.KEYPOINT_ELBOW_LEFT
        else:
            wrist_idx = config.KEYPOINT_WRIST_RIGHT
            elbow_idx = config.KEYPOINT_ELBOW_RIGHT

        wrist = self._visible_keypoint(keypoints, wrist_idx)
        elbow = self._visible_keypoint(keypoints, elbow_idx)

        wrist_contact = wrist is not None and self._point_in_or_near_roi(
            lock_roi_name, wrist, config.HAND_LOCK_PROXIMITY_PIXELS
        )
        elbow_contact = elbow is not None and self._point_in_or_near_roi(
            lock_roi_name, elbow, config.ELBOW_LOCK_PROXIMITY_PIXELS
        )
        return wrist_contact or elbow_contact

    def _left_right_keypoints_in_video_order(self, keypoints: np.ndarray) -> bool:
        """
        Door-facing/back-to-camera check requested for this camera:
        left elbow/ankle should appear left in the video, and right elbow/ankle
        should appear right in the video.
        """
        required_pairs = [
            (config.KEYPOINT_ELBOW_LEFT, config.KEYPOINT_ELBOW_RIGHT),
            (config.KEYPOINT_ANKLE_LEFT, config.KEYPOINT_ANKLE_RIGHT),
        ]
        for left_idx, right_idx in required_pairs:
            left = self._visible_keypoint(keypoints, left_idx)
            right = self._visible_keypoint(keypoints, right_idx)
            if left is None or right is None:
                return False
            if right[0] <= left[0] + config.LEFT_RIGHT_ORDER_MIN_PIXELS:
                return False
        return True

    def _visible_keypoint(self, keypoints: np.ndarray, idx: int) -> Optional[Tuple[float, float]]:
        if idx >= len(keypoints):
            return None
        x, y, conf = keypoints[idx]
        if conf < config.ARM_KEYPOINT_CONFIDENCE_THRESHOLD:
            return None
        return float(x), float(y)

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

        min_dist = self._min_distance_to_polygon("LOCK_A_ROI", head_pos)
        if min_dist > config.HEAD_TO_LOCKER_A_MAX_PIXELS:
            return False

        if not (self._side_contacts_any_lock(keypoints, "left") and
                self._side_contacts_any_lock(keypoints, "right")):
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

    def get_unlocker_by_tag(self, tag: str) -> Optional[int]:
        """Get current primary ID for unlocker tag."""
        ids = self.all_unlocker_ids.get(tag, set())
        if ids:
            return min(ids)  # Return earliest assigned ID as primary
        return None

    def get_all_ids_for_tag(self, tag: str) -> Set[int]:
        """Get all IDs ever assigned to this unlocker."""
        return self.all_unlocker_ids.get(tag, set()).copy()

    def is_unlocker_tracked(self, slot: str, tracked_persons: Dict[int, Dict]) -> bool:
        """
        Check if unlocker is currently tracked (by ANY of their IDs).
        Enables tracking continuity across ID changes.
        """
        tag = f"P{1 if slot == 'a' else 2}_unlocker"
        unlocker_ids = self.all_unlocker_ids.get(tag, set())
        return any(track_id in tracked_persons for track_id in unlocker_ids)

    def try_reassign_unlocker_id(self, slot: str, tracked_persons: Dict[int, Dict], anchor: Optional[Tuple]) -> Optional[int]:
        """
        If unlocker's current ID is lost, find new ID by anchor matching.
        Assign new ID same unlocker tag for continuity.

        Returns: New track_id if match found, else None
        """
        tag = f"P{1 if slot == 'a' else 2}_unlocker"
        current_id = self.session.get(f"id_{slot}")

        # Already have valid current ID tracked
        if current_id is not None and current_id in tracked_persons:
            return None

        # Find new ID matching anchor
        if anchor is None:
            return None

        best_match = None
        best_dist = config.UNLOCKER_ANCHOR_MATCH_PIXELS

        for track_id, person in tracked_persons.items():
            # Skip if already tagged as different unlocker
            if track_id in self.unlocker_tags and self.unlocker_tags[track_id] != tag:
                continue
            # Skip if already our unlocker
            if track_id in self.all_unlocker_ids[tag]:
                continue

            bbox = person.get("bbox")
            if bbox is None:
                continue

            center = self._bbox_bottom_center(bbox)
            dist = np.linalg.norm(np.array(center) - np.array(anchor))

            if dist < best_dist:
                best_dist = dist
                best_match = track_id

        if best_match is not None:
            self.assign_unlocker_tag(best_match, slot)
            print(f"[REASSIGN] P{1 if slot == 'a' else 2} ID {current_id} lost, reassigned to ID {best_match} (dist={best_dist:.1f})")
            return best_match

        return None

    # ================================================================
    # AUTHORIZATION
    # ================================================================
    def check_authorization(self) -> Dict:
        id_a = self.session["id_a"]
        id_b = self.session["id_b"]
        lock_a_auth = id_a is not None and self.session["timer_a_frames"] >= self.min_unlock_frames
        lock_b_auth = id_b is not None and self.session["timer_b_frames"] >= self.min_unlock_frames

        if id_a is not None and id_b is not None and id_a == id_b:
            return {
                "authorized": False,
                "lock_a_authorized": lock_a_auth,
                "lock_b_authorized": lock_b_auth,
                "violation_type": "SAME_ID",
            }

        authorized = lock_a_auth and lock_b_auth and id_a != id_b

        if authorized and self.authorized_session_buffer is None:
            self.authorized_session_buffer = {
                "id_a": id_a,
                "id_b": id_b,
                "timer_a_seconds": self.session["timer_a_seconds"],
                "timer_b_seconds": self.session["timer_b_seconds"],
                "sequence_state": self.session["sequence_state"],
            }
            print("[AUTH] Session buffer saved: P1 and P2 verified")

        return {
            "authorized": authorized,
            "lock_a_authorized": lock_a_auth,
            "lock_b_authorized": lock_b_auth,
            "violation_type": None if authorized else "INCOMPLETE",
        }

    def verified_unlockers_in_interaction_zone(self, tracked_persons: Dict[int, Dict]) -> bool:
        """Return True only when verified P1 and P2 are both currently visible in INTERACTION_ZONE."""
        for slot in ("a", "b"):
            track_id = self.session.get(f"id_{slot}")
            if track_id is None or track_id not in tracked_persons:
                return False

            keypoints = tracked_persons[track_id].get("keypoints")
            if keypoints is None:
                return False

            head_pos = self._get_head_position(keypoints)
            if head_pos is None:
                return False

            if not self.roi_manager.point_in_roi("INTERACTION_ZONE", head_pos[0], head_pos[1]):
                return False

        return True

    def check_door_violation(self) -> Optional[str]:
        if self.authorized_session_buffer is None:
            return "LONE_WOLF"

        buf = self.authorized_session_buffer
        if buf["id_a"] != buf["id_b"] and buf["id_a"] is not None and buf["id_b"] is not None:
            return None
        return "LONE_WOLF"

    def get_session(self) -> Dict:
        return self.session.copy()

    def reset_session(self):
        self._reset_all_timers()
        self.authorized_session_buffer = None
        self.current_frame_count += 1
