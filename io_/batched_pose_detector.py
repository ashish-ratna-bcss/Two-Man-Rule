"""
Batched Pose Detector — Wrapper around GPUBatchCoordinator for drop-in compatibility.

Presents the same detect(frame) interface as standalone PoseDetector, but routes
inference through a shared GPU batch coordinator for zero-latency multi-stream serving.
"""

from typing import List, Dict
import numpy as np
from io_.gpu_batch_coordinator import GPUBatchCoordinator
import config


class BatchedPoseDetector:
    """
    Drop-in replacement for PoseDetector when batch scheduler is enabled.
    All instances share one GPUBatchCoordinator singleton.
    """

    _coordinator = None
    _coordinator_lock = __import__("threading").Lock()

    def __init__(
        self,
        model_path: str = None,
        device: str = "auto",
        half: bool = True,
        stream_id: str = "default",
        batch_size: int = None,
        cadence_ms: float = None,
        preallocate_mb: float = None,
    ):
        """
        Initialize a batched detector. Coordinator is created on first use.

        Args:
            model_path: YOLO model path
            device: CUDA device (e.g., 'cuda', 'cpu', 'auto')
            half: Use FP16 if True and CUDA available
            stream_id: Unique identifier for this stream
            batch_size: Override BATCH_SIZE from config
            cadence_ms: Override BATCH_INFERENCE_CADENCE_MS from config
            preallocate_mb: Override BATCH_GPU_PREALLOCATE_MB from config
        """
        self.model_path = model_path or config.YOLO_POSE_MODEL
        self.device = device if device != "auto" else ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self.half = half
        self.stream_id = stream_id
        self.last_results = None

        batch_size = batch_size or config.BATCH_SIZE
        cadence_ms = cadence_ms or config.BATCH_INFERENCE_CADENCE_MS
        preallocate_mb = preallocate_mb or config.BATCH_GPU_PREALLOCATE_MB

        # Lazy initialize coordinator on first detector
        with BatchedPoseDetector._coordinator_lock:
            if BatchedPoseDetector._coordinator is None:
                BatchedPoseDetector._coordinator = GPUBatchCoordinator(
                    model_path=self.model_path,
                    device=self.device,
                    half=self.half,
                    batch_size=batch_size,
                    cadence_ms=cadence_ms,
                    preallocate_mb=preallocate_mb,
                )
                BatchedPoseDetector._coordinator.register_stream(
                    stream_id,
                    max_input_queue=config.BATCH_INPUT_QUEUE_SIZE,
                    max_output_queue=config.BATCH_OUTPUT_QUEUE_SIZE,
                )
                BatchedPoseDetector._coordinator.start()
            else:
                # Register this stream with the already-running coordinator
                BatchedPoseDetector._coordinator.register_stream(
                    stream_id,
                    max_input_queue=config.BATCH_INPUT_QUEUE_SIZE,
                    max_output_queue=config.BATCH_OUTPUT_QUEUE_SIZE,
                )

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Queue a frame for inference and retrieve results (blocking).

        This is the drop-in replacement for PoseDetector.detect().
        """
        # Enqueue frame
        self._coordinator.enqueue_frame(self.stream_id, frame)

        # Retrieve results (blocking, with timeout)
        results = self._coordinator.get_results(self.stream_id, timeout=2.0)
        if results is None:
            # Timeout — return None so main.py LKG path activates (matches PoseDetector contract)
            return None

        self.last_results = results
        return results

    @classmethod
    def get_coordinator(cls) -> GPUBatchCoordinator:
        """Get the shared coordinator instance."""
        return cls._coordinator

    @classmethod
    def stop_coordinator(cls):
        """Stop the shared coordinator and cleanup."""
        with cls._coordinator_lock:
            if cls._coordinator:
                cls._coordinator.stop()
                cls._coordinator = None
