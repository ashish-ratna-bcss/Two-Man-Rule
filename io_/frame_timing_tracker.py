"""
Simple frame timing tracker for per-frame analysis.
Records arrival time, inference duration, GPU memory usage and detection quality.
Exports CSV to `logs/frame_timing.csv` when `export_csv()` is called.
"""
from dataclasses import dataclass, asdict
from typing import Optional, List
import time
import csv
from pathlib import Path

@dataclass
class FrameTimingEvent:
    frame_idx: int
    camera_id: str
    t_arrival: float
    t_inference_start: Optional[float] = None
    t_inference_end: Optional[float] = None
    gpu_mem_before_mb: Optional[float] = None
    gpu_mem_after_mb: Optional[float] = None
    num_persons: Optional[int] = None
    avg_confidence: Optional[float] = None
    wrist_keypoints: Optional[int] = None

class FrameTimingTracker:
    _instance = None

    def __init__(self):
        self.events: List[FrameTimingEvent] = []

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record_event(self, event: FrameTimingEvent):
        self.events.append(event)

    def export_csv(self, path: str = "logs/frame_timing.csv"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.events[0]).keys()) if self.events else [
                'frame_idx','camera_id','t_arrival','t_inference_start','t_inference_end',
                'gpu_mem_before_mb','gpu_mem_after_mb','num_persons','avg_confidence','wrist_keypoints'
            ])
            writer.writeheader()
            for e in self.events:
                writer.writerow(asdict(e))

    def clear(self):
        self.events.clear()
