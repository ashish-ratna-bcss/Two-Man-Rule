# SAME_ID Return Timer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require P1 to sustain lock contact for `MIN_UNLOCK_SECONDS` (5s) after returning to the door before a SAME_ID violation is fired, eliminating false-positive alerts from brief re-approaches.

**Architecture:** Add two session keys (`same_id_return_timer_frames`, `same_id_return_grace_frames`) and replace the immediate violation trigger at `state_machine.py:315–319` with a frame-accumulating timer that mirrors the normal unlock verification pattern.

**Tech Stack:** Python, pytest

---

### Task 1: Add session keys to `config.py`

**Files:**
- Modify: `config.py:238–255` (`create_session`)

- [ ] **Step 1: Add two keys to `create_session()`**

In `config.py`, inside the dict returned by `create_session()`, add after `"violation_type": None,`:

```python
"same_id_return_timer_frames": 0,
"same_id_return_grace_frames": 0,
```

Full updated dict (lines 238–255 region):

```python
def create_session():
    """Create a fresh session state dict."""
    return {
        "sequence_state": "WAITING_FOR_FIRST_UNLOCKER",
        "candidate_a": None,
        "candidate_b": None,
        "id_a": None,
        "id_b": None,
        "timer_a_frames": 0,
        "timer_b_frames": 0,
        "timer_a_seconds": 0.0,
        "timer_b_seconds": 0.0,
        "grace_buffer_a": 0,
        "grace_buffer_b": 0,
        "improper_positioning": None,
        "violation_type": None,
        "captured_violations": [],
        "same_id_return_timer_frames": 0,
        "same_id_return_grace_frames": 0,
    }
```

- [ ] **Step 2: Verify import still works**

```bash
cd "/home/ashish-ratna/PMJ/Two-Man Rule"
python -c "import config; s = config.create_session(); print(s['same_id_return_timer_frames'], s['same_id_return_grace_frames'])"
```

Expected output: `0 0`

---

### Task 2: Write failing tests

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_same_id_return_timer.py`

- [ ] **Step 1: Create tests directory and empty `__init__.py`**

```bash
mkdir -p "/home/ashish-ratna/PMJ/Two-Man Rule/tests"
touch "/home/ashish-ratna/PMJ/Two-Man Rule/tests/__init__.py"
```

- [ ] **Step 2: Write the test file**

Create `tests/test_same_id_return_timer.py`:

```python
"""Tests for SAME_ID return timer — P1 must sustain lock contact for min_unlock_frames
before a violation is fired."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch
import config
from logic.state_machine import DualAuthStateMachine


def _make_sm(fps=10):
    """StateMachine at 10 FPS so min_unlock_frames = 50 (5s × 10fps)."""
    roi = MagicMock()
    roi.point_in_roi.return_value = False
    roi.get_roi.return_value = None
    sm = DualAuthStateMachine(roi_manager=roi, fps=fps)
    return sm


def _returning_pose(head_in_door=True, has_lock_contact=True):
    """Minimal pose_results entry simulating P1 at door with lock contact."""
    return {
        "qualified": False,
        "head_in_interaction": False,
        "feet_in_standing": False,
        "waist_near_door": False,
        "ear_order_correct": True,
        "shoulder_order_correct": True,
        "in_locks_roi": False,
        "arms_raised": False,
        "left_right_order": True,
        "has_lock_contact": has_lock_contact,
        "head_in_door": head_in_door,
        "shoulders_in_door": False,
        "anchor": (100, 100),
    }


def _call_with_pose(sm, track_id, pose_override, frame_step=1):
    """Call update_timers with _evaluate_unlock_pose patched to return pose_override."""
    tracked = {track_id: {"keypoints": None, "bbox": None}}
    with patch.object(sm, "_evaluate_unlock_pose", return_value=pose_override):
        sm.update_timers(tracked, frame_step=frame_step)


def _seed_p1_returned(sm, track_id=1):
    """Pre-seed session: P1 verified and has already left the zone once."""
    sm.session["id_a"] = track_id
    sm.session["id_b"] = None
    sm.session["id_a_left_zone"] = True
    sm.verified_anchors["a"] = (100, 100)


def test_no_violation_before_threshold():
    """Timer accumulates but no violation fires below min_unlock_frames."""
    sm = _make_sm(fps=10)  # min_unlock_frames = 50
    _seed_p1_returned(sm, track_id=1)

    pose = _returning_pose(head_in_door=True, has_lock_contact=True)
    for _ in range(49):  # one frame short of threshold
        _call_with_pose(sm, track_id=1, pose_override=pose)

    assert sm.session["violation_type"] is None
    assert sm.session["same_id_return_timer_frames"] == 49


def test_violation_fires_at_threshold():
    """Violation fires exactly when timer reaches min_unlock_frames."""
    sm = _make_sm(fps=10)
    _seed_p1_returned(sm, track_id=1)

    pose = _returning_pose(head_in_door=True, has_lock_contact=True)
    for _ in range(50):  # exactly min_unlock_frames
        _call_with_pose(sm, track_id=1, pose_override=pose)

    assert sm.session["violation_type"] == "SAME_ID"


def test_timer_pauses_during_grace_not_reset():
    """Brief contact loss within GRACE_BUFFER_FRAMES pauses timer — does not reset it."""
    sm = _make_sm(fps=10)
    _seed_p1_returned(sm, track_id=1)

    contact_pose = _returning_pose(head_in_door=True, has_lock_contact=True)
    no_contact_pose = _returning_pose(head_in_door=True, has_lock_contact=False)

    # Accumulate 20 frames
    for _ in range(20):
        _call_with_pose(sm, track_id=1, pose_override=contact_pose)

    assert sm.session["same_id_return_timer_frames"] == 20

    # Drop contact for fewer frames than GRACE_BUFFER_FRAMES (15 at 30fps; scaled here)
    grace = config.GRACE_BUFFER_FRAMES
    for _ in range(grace - 1):
        _call_with_pose(sm, track_id=1, pose_override=no_contact_pose)

    # Timer must still be 20 (paused, not reset)
    assert sm.session["same_id_return_timer_frames"] == 20
    assert sm.session["violation_type"] is None


def test_timer_resets_after_grace_expires():
    """If contact loss exceeds grace, timer resets to zero."""
    sm = _make_sm(fps=10)
    _seed_p1_returned(sm, track_id=1)

    contact_pose = _returning_pose(head_in_door=True, has_lock_contact=True)
    no_contact_pose = _returning_pose(head_in_door=True, has_lock_contact=False)

    for _ in range(20):
        _call_with_pose(sm, track_id=1, pose_override=contact_pose)

    grace = config.GRACE_BUFFER_FRAMES
    for _ in range(grace + 1):  # one beyond grace
        _call_with_pose(sm, track_id=1, pose_override=no_contact_pose)

    assert sm.session["same_id_return_timer_frames"] == 0
    assert sm.session["violation_type"] is None


def test_no_violation_when_no_lock_contact():
    """No violation even after many frames if lock contact never made."""
    sm = _make_sm(fps=10)
    _seed_p1_returned(sm, track_id=1)

    pose = _returning_pose(head_in_door=True, has_lock_contact=False)
    for _ in range(100):
        _call_with_pose(sm, track_id=1, pose_override=pose)

    assert sm.session["violation_type"] is None
    assert sm.session["same_id_return_timer_frames"] == 0


def test_no_violation_when_id_b_already_set():
    """SAME_ID check skipped entirely if P2 already verified."""
    sm = _make_sm(fps=10)
    sm.session["id_a"] = 1
    sm.session["id_b"] = 2  # P2 already done — skip check
    sm.session["id_a_left_zone"] = True

    pose = _returning_pose(head_in_door=True, has_lock_contact=True)
    for _ in range(100):
        _call_with_pose(sm, track_id=1, pose_override=pose)

    assert sm.session["violation_type"] is None
```

- [ ] **Step 3: Run tests to confirm they all FAIL**

```bash
cd "/home/ashish-ratna/PMJ/Two-Man Rule"
python -m pytest tests/test_same_id_return_timer.py -v
```

Expected: most tests FAIL (currently violation fires immediately, not after timer).

---

### Task 3: Implement the timer logic in `state_machine.py`

**Files:**
- Modify: `logic/state_machine.py:315–323`

- [ ] **Step 1: Replace the immediate violation block with timer logic**

In `logic/state_machine.py`, replace lines 315–323:

```python
                if self.session["id_a_left_zone"]:
                    # Violation: same ID returns to door (head OR shoulder) and touches locks
                    if (res.get("head_in_door") or res.get("shoulders_in_door")) and res.get("has_lock_contact"):
                        print(f"[VIOLATION] P1 (ID {id_a}) returned to DOOR_ROI and interacting with LOCKS_ROI!")
                        self.session["violation_type"] = "SAME_ID"
                    else:
                        # Debug if they are interacting but not triggering violation
                        if res.get("has_lock_contact"):
                             print(f"[DEBUG] P1 (ID {id_a}) interacting with locks. head_in_door={res.get('head_in_door')} shoulders_in_door={res.get('shoulders_in_door')}")
```

With:

```python
                if self.session["id_a_left_zone"]:
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
                            # timer pauses — holds accumulated value through brief dropout
                        elif timer > 0:
                            # grace expired — P1 clearly disengaged, reset
                            self.session["same_id_return_timer_frames"] = 0
                            self.session["same_id_return_grace_frames"] = 0
                        if res.get("has_lock_contact"):
                            print(f"[DEBUG] P1 (ID {id_a}) lock contact but not at door. head_in_door={res.get('head_in_door')} shoulders_in_door={res.get('shoulders_in_door')}")
```

- [ ] **Step 2: Run tests — all must pass**

```bash
cd "/home/ashish-ratna/PMJ/Two-Man Rule"
python -m pytest tests/test_same_id_return_timer.py -v
```

Expected output:
```
PASSED tests/test_same_id_return_timer.py::test_no_violation_before_threshold
PASSED tests/test_same_id_return_timer.py::test_violation_fires_at_threshold
PASSED tests/test_same_id_return_timer.py::test_timer_pauses_during_grace_not_reset
PASSED tests/test_same_id_return_timer.py::test_timer_resets_after_grace_expires
PASSED tests/test_same_id_return_timer.py::test_no_violation_when_no_lock_contact
PASSED tests/test_same_id_return_timer.py::test_no_violation_when_id_b_already_set
6 passed
```

- [ ] **Step 3: Commit**

```bash
cd "/home/ashish-ratna/PMJ/Two-Man Rule"
git add config.py logic/state_machine.py tests/
git commit -m "fix: require 5s sustained interaction before SAME_ID violation fires

Previously the violation triggered instantly when P1 re-approached
the door after leaving. Now requires min_unlock_frames of continuous
lock contact (with grace buffer), matching the normal unlock threshold.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```
