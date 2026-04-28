# logic/person_memory.py
import json
import os
from datetime import datetime
from typing import Dict, Optional, List
import config


class PersonMemory:
    """Persistent memory for verified persons who qualified for unlock."""

    def __init__(self, session_id: str):
        """
        Init person memory with session ID.

        Args:
            session_id: Unique session identifier (timestamp-based)
        """
        self.session_id = session_id
        self.memory_dir = os.path.join(config.LOG_DIR, "person_memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.memory_file = os.path.join(self.memory_dir, f"verified_{session_id}.json")
        self.verified_persons = {}  # track_id -> person_data

    def record_verified_person(self, slot: str, track_id: int, person_data: Dict) -> None:
        """
        Record person who qualified for unlock.

        Args:
            slot: 'a' or 'b' (P1 or P2)
            track_id: Assigned ByteTrack ID
            person_data: Full detection data (bbox, keypoints, confidence)
        """
        person_entry = {
            "slot": slot,
            "track_id": track_id,
            "timestamp": datetime.now().isoformat(),
            "bbox": person_data.get("bbox"),
            "confidence": float(person_data.get("confidence", 0.0)),
            "keypoints_shape": list(person_data.get("keypoints", []).shape) if hasattr(person_data.get("keypoints"), "shape") else None,
        }
        self.verified_persons[track_id] = person_entry
        self._save_to_disk()
        print(f"[MEMORY] P{1 if slot == 'a' else 2} (ID {track_id}) recorded: {person_entry['timestamp']}")

    def record_door_state(self, door_opened: bool, reason: str = "") -> None:
        """Record final door state after verification."""
        self.verified_persons["_door_result"] = {
            "timestamp": datetime.now().isoformat(),
            "door_opened": door_opened,
            "reason": reason,
        }
        self._save_to_disk()
        print(f"[MEMORY] Door state recorded: opened={door_opened}, reason='{reason}'")

    def record_authorization_result(self, authorized: bool, details: Dict) -> None:
        """Record final authorization check result."""
        self.verified_persons["_auth_result"] = {
            "timestamp": datetime.now().isoformat(),
            "authorized": authorized,
            "details": details,
        }
        self._save_to_disk()
        print(f"[MEMORY] Auth result recorded: authorized={authorized}")

    def get_verified_ids(self) -> Dict[str, int]:
        """Get P1/P2 assigned IDs."""
        result = {}
        for track_id, entry in self.verified_persons.items():
            if isinstance(entry, dict) and "slot" in entry:
                result[entry["slot"]] = track_id
        return result

    def get_memory_summary(self) -> Dict:
        """Get summary of verified persons and results."""
        summary = {
            "session_id": self.session_id,
            "verified_persons": {},
            "door_result": None,
            "auth_result": None,
        }

        for track_id, entry in self.verified_persons.items():
            if isinstance(entry, dict):
                if track_id == "_door_result":
                    summary["door_result"] = entry
                elif track_id == "_auth_result":
                    summary["auth_result"] = entry
                elif "slot" in entry:
                    summary["verified_persons"][f"P{1 if entry['slot'] == 'a' else 2}"] = entry

        return summary

    def _save_to_disk(self) -> None:
        """Persist memory to JSON file."""
        try:
            with open(self.memory_file, "w") as f:
                json.dump(self.verified_persons, f, indent=2)
        except Exception as e:
            print(f"[MEMORY] Save failed: {e}")

    def load_from_disk(self) -> bool:
        """Load existing memory from disk."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    self.verified_persons = json.load(f)
                print(f"[MEMORY] Loaded from {self.memory_file}")
                return True
            except Exception as e:
                print(f"[MEMORY] Load failed: {e}")
        return False

    def clear(self) -> None:
        """Clear memory."""
        self.verified_persons = {}
        self._save_to_disk()
        print(f"[MEMORY] Cleared")
