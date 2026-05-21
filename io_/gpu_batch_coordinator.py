"""
GPU Batch Coordinator — Zero-latency multi-stream inference.

Design:
  - All streams queue frames to a shared batch buffer (no contention).
  - GPU worker runs on a FIXED cadence (e.g., 30ms for 30 FPS).
  - Results routed back to per-stream response queues.
  - Per-stream frame-drop tracking (oldest frame dropped if queue is full).
  - GPU memory pre-allocated once at startup.

Benefits:
  - Adding streams does NOT increase per-frame latency.
  - GPU memory usage is predictable and fixed.
  - Frame drops are per-stream, not global.
  - Inference throughput is bounded by GPU compute, not orchestration.
"""

import queue
import time
import threading
from typing import Dict, List, Optional, Any
import numpy as np
import torch

import config


class StreamFrameBuffer:
    """Per-stream input queue with automatic frame drop on overflow."""

    def __init__(self, stream_id: str, max_queue_size: int = 5):
        self.stream_id = stream_id
        self.frame_queue: queue.Queue = queue.Queue(maxsize=0)  # Unbounded
        self.max_queue_size = max_queue_size
        self.dropped_frames = 0

    def put_frame(self, frame: np.ndarray) -> bool:
        """
        Queue a frame. If queue size exceeds max_queue_size, drop oldest frame.
        Returns True if queued, False if dropped.
        """
        if self.frame_queue.qsize() < self.max_queue_size:
            self.frame_queue.put(frame, block=False)
            return True
        else:
            # Queue is full; drop oldest frame and queue the new one
            try:
                self.frame_queue.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass
            self.frame_queue.put(frame, block=False)
            return False

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get the oldest queued frame, or None if none available."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self) -> int:
        return self.frame_queue.qsize()

    def clear(self):
        while True:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break


class StreamResponseQueue:
    """Per-stream output queue for inference results."""

    def __init__(self, stream_id: str, max_queue_size: int = 10):
        self.stream_id = stream_id
        self.response_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.max_queue_size = max_queue_size
        self.dropped_results = 0

    def put_result(self, result: Any) -> bool:
        """
        Queue an inference result. If full, drop oldest result.
        Returns True if queued, False if oldest was dropped.
        """
        try:
            self.response_queue.put(result, block=False)
            return True
        except queue.Full:
            # Drop oldest result
            try:
                self.response_queue.get_nowait()
                self.dropped_results += 1
            except queue.Empty:
                pass
            self.response_queue.put(result, block=False)
            return False

    def get_result(self, timeout: float = 2.0) -> Optional[Any]:
        """Get an inference result, or None if timeout."""
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self) -> int:
        return self.response_queue.qsize()


class GPUBatchCoordinator:
    """
    Manages all-streams-to-GPU coordination with fixed cadence and pre-allocated memory.

    Usage:
      coordinator = GPUBatchCoordinator(model_path, device, num_streams)
      for stream_id in stream_list:
          coordinator.register_stream(stream_id)
      coordinator.start()
      # Main loop:
      for each frame from stream X:
          coordinator.enqueue_frame(stream_id=X, frame=frame_data)
      # When ready to get results:
      results = coordinator.get_results(stream_id=X, timeout=2.0)
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        half: bool = True,
        batch_size: int = 4,
        cadence_ms: float = 33.0,  # ~30 FPS
        preallocate_mb: float = 1024.0,
    ):
        self.model_path = model_path or config.YOLO_POSE_MODEL
        self.device = device if torch.cuda.is_available() else "cpu"
        self.half = half and self.device == "cuda"
        self.batch_size = max(int(batch_size), 1)
        self.cadence_ms = float(cadence_ms)
        self.preallocate_mb = float(preallocate_mb)

        self.stream_buffers: Dict[str, StreamFrameBuffer] = {}
        self.stream_responses: Dict[str, StreamResponseQueue] = {}
        self.model = None
        self.running = False
        self.worker_thread = None
        self.stats = {
            "frames_processed": 0,
            "batches_run": 0,
            "inference_time_ms": 0.0,
        }

    def register_stream(self, stream_id: str, max_input_queue: int = 5, max_output_queue: int = 10):
        """Register a stream before starting the coordinator."""
        if stream_id not in self.stream_buffers:
            self.stream_buffers[stream_id] = StreamFrameBuffer(stream_id, max_queue_size=max_input_queue)
            self.stream_responses[stream_id] = StreamResponseQueue(stream_id, max_queue_size=max_output_queue)

    def start(self):
        """Load GPU model, pre-allocate memory, and start the worker thread."""
        if self.running:
            print("[GPUBatchCoordinator] Already running.")
            return

        print(f"[GPUBatchCoordinator] Loading model on {self.device}...")
        from ultralytics import YOLO

        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        # Pre-allocate GPU memory
        if self.device == "cuda":
            try:
                dummy = torch.zeros(
                    (self.batch_size, 3, 640, 640),
                    device=self.device,
                    dtype=torch.float16 if self.half else torch.float32,
                )
                _ = self.model(dummy, verbose=False)
                torch.cuda.synchronize()
                print(f"[GPUBatchCoordinator] GPU memory pre-allocated for batch size {self.batch_size}.")
                del dummy
            except Exception as e:
                print(f"[GPUBatchCoordinator] WARNING: Pre-allocation failed: {e}")

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print(f"[GPUBatchCoordinator] Worker thread started (cadence={self.cadence_ms}ms).")

    def stop(self):
        """Stop the worker thread and clean up."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        if self.model:
            self.model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
        print("[GPUBatchCoordinator] Stopped.")

    def enqueue_frame(self, stream_id: str, frame: np.ndarray) -> bool:
        """Queue a frame from a stream. Returns True if queued, False if dropped."""
        if stream_id not in self.stream_buffers:
            print(f"[GPUBatchCoordinator] WARNING: Stream {stream_id} not registered.")
            return False
        return self.stream_buffers[stream_id].put_frame(frame)

    def get_results(self, stream_id: str, timeout: float = 2.0) -> Optional[List[Dict]]:
        """Retrieve inference results for a stream."""
        if stream_id not in self.stream_responses:
            return None
        return self.stream_responses[stream_id].get_result(timeout=timeout)

    def get_stats(self) -> Dict[str, Any]:
        """Return current statistics."""
        stats = dict(self.stats)
        stats["registered_streams"] = len(self.stream_buffers)
        for sid, buf in self.stream_buffers.items():
            stats[f"{sid}_input_queue_size"] = buf.qsize()
            stats[f"{sid}_dropped_frames"] = buf.dropped_frames
        for sid, resp in self.stream_responses.items():
            stats[f"{sid}_output_queue_size"] = resp.qsize()
            stats[f"{sid}_dropped_results"] = resp.dropped_results
        return stats

    def _worker_loop(self):
        """Main inference loop: run batches on fixed cadence."""
        print("[GPUBatchCoordinator] Worker loop started.")
        loop_start = time.perf_counter()

        try:
            while self.running:
                batch_deadline = loop_start + (self.cadence_ms / 1000.0)

                # Collect frames for a batch
                stream_ids_in_batch = []
                batch_frames = []
                for stream_id in list(self.stream_buffers.keys()):
                    if len(batch_frames) >= self.batch_size:
                        break
                    frame = self.stream_buffers[stream_id].get_frame(timeout=0.001)
                    if frame is not None:
                        stream_ids_in_batch.append(stream_id)
                        batch_frames.append(frame)

                if not batch_frames:
                    # No frames available; sleep briefly and loop
                    time.sleep(0.001)
                    loop_start = time.perf_counter()
                    continue

                # Run inference
                try:
                    t_inf_start = time.perf_counter()
                    with torch.inference_mode():
                        results = self.model(batch_frames, conf=0.5, verbose=False, device=self.device, half=self.half)
                    t_inf_end = time.perf_counter()
                    self.stats["inference_time_ms"] = (t_inf_end - t_inf_start) * 1000.0
                    self.stats["batches_run"] += 1
                    self.stats["frames_processed"] += len(batch_frames)

                    # Route results back to per-stream queues
                    for stream_id, yolo_result in zip(stream_ids_in_batch, results):
                        detections = self._process_result(yolo_result)
                        self.stream_responses[stream_id].put_result(detections)

                except torch.cuda.OutOfMemoryError:
                    print("[GPUBatchCoordinator] CUDA OOM! Clearing cache and retrying.")
                    torch.cuda.empty_cache()
                    # Return empty results for this batch
                    for stream_id in stream_ids_in_batch:
                        self.stream_responses[stream_id].put_result([])
                except Exception as e:
                    print(f"[GPUBatchCoordinator] Inference error: {e}")
                    for stream_id in stream_ids_in_batch:
                        self.stream_responses[stream_id].put_result([])

                # Sleep until next batch deadline
                elapsed = time.perf_counter() - loop_start
                remaining_ms = (self.cadence_ms / 1000.0) - elapsed
                if remaining_ms > 0:
                    time.sleep(remaining_ms * 0.95)  # Sleep 95% to account for scheduling jitter

                loop_start = time.perf_counter()

        except Exception as e:
            print(f"[GPUBatchCoordinator] Worker thread exception: {e}")
        finally:
            print("[GPUBatchCoordinator] Worker loop stopped.")

    def _process_result(self, yolo_result) -> List[Dict]:
        """Convert YOLO result to detection list."""
        detections = []
        if yolo_result.boxes is not None and yolo_result.keypoints is not None:
            boxes_xyxy = yolo_result.boxes.xyxy.cpu().numpy()
            boxes_conf = yolo_result.boxes.conf.cpu().numpy()
            kpts_xy = yolo_result.keypoints.xy.cpu().numpy()
            kpts_conf = yolo_result.keypoints.conf.cpu().numpy()

            for i in range(len(boxes_xyxy)):
                kpt_array = np.hstack([kpts_xy[i], kpts_conf[i].reshape(-1, 1)])
                detections.append({
                    "bbox": boxes_xyxy[i],
                    "confidence": float(boxes_conf[i]),
                    "keypoints": kpt_array,
                    "keypoint_names": [
                        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist", "left_hip", "right_hip",
                        "left_knee", "right_knee", "left_ankle", "right_ankle",
                    ],
                })
        return detections
