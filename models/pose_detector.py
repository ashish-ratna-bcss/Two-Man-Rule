# models/pose_detector.py
from ultralytics import YOLO
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import config
import time
import os
import multiprocessing as mp
from multiprocessing.managers import BaseManager, DictProxy
from multiprocessing import shared_memory

class InferenceManager(BaseManager): pass

def get_shared_queues(address=('127.0.0.1', 50000), authkey=b'pmj_auth'):
    """Connect to a running InferenceManager and return the shared queues.
    
    Architecture: single request queue + single response queue.
    Workers tag requests with their client_id; server echoes the tag back.
    Workers poll the shared response queue and discard messages not for them.
    This avoids storing mp.Queue objects in a Manager dict (which is not picklable).
    """
    InferenceManager.register('get_request_queue')
    InferenceManager.register('get_response_queue')
    InferenceManager.register('get_shared_memory_config', proxytype=DictProxy)
    manager = InferenceManager(address=address, authkey=authkey)
    try:
        manager.connect()
        return (
            manager.get_request_queue(),
            manager.get_response_queue(),
            manager.get_shared_memory_config(),
        )
    except Exception as e:
        print(f"[InferenceManager] Could not connect to shared server: {e}")
        return None, None, None

def start_inference_manager(request_q, response_q, shm_config, address=('127.0.0.1', 50000), authkey=b'pmj_auth'):
    """Start a manager server to host the shared queues and shm info."""
    InferenceManager.register('get_request_queue', callable=lambda: request_q)
    InferenceManager.register('get_response_queue', callable=lambda: response_q)
    InferenceManager.register('get_shared_memory_config', callable=lambda: shm_config, proxytype=DictProxy)
    manager = InferenceManager(address=address, authkey=authkey)
    server = manager.get_server()
    return server

class PoseDetector:
    """YOLOv8-Pose wrapper optimized for production CUDA environments."""

    def __init__(self, model_path: str = None, device: str = "auto", half: bool = True, shared_mode: bool = None):
        """
        Initialize PoseDetector.
        If shared_mode is True, it connects to a central _InferenceServer instead of loading its own model.
        """
        self.model_path = model_path or config.YOLO_POSE_MODEL
        self.device_request = device
        self.half_request = half
        
        # Use config default if not specified
        self.shared_mode = shared_mode if shared_mode is not None else getattr(config, "SHARED_INFERENCE_ENABLED", False)
        
        self.model = None
        self.device = None
        self.use_half = False
        self.last_results = None
        
        # Client-side state for shared mode
        self.client_id = f"client_{os.getpid()}_{id(self)}" if self.shared_mode else None
        self._request_queue = None
        self._response_queue = None
        self._shm = None
        self._shm_buf = None
        self._shm_slot_idx = -1
        self._shm_slot_size = 0
        
        if not self.shared_mode:
            print(f"[PoseDetector] Initialized in STANDALONE mode (Lazy Load enabled)")
        else:
            print(f"[PoseDetector] Initialized in SHARED mode (Client ID: {self.client_id})")

    def _ensure_model_loaded(self):
        """Lazy load the model only when first inference is requested."""
        if self.model is not None or self.shared_mode:
            return

        print(f"[PoseDetector] Lazy loading model: {self.model_path}")
        if self.device_request == "auto":
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = self.device_request
            if self.device == "cuda" and not torch.cuda.is_available():
                print("[PoseDetector] CUDA requested but unavailable; falling back to CPU")
                self.device = "cpu"

        self.use_half = bool(self.half_request and self.device == "cuda")
        
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(config, "MAX_PROCESS_VRAM_FRACTION") and config.MAX_PROCESS_VRAM_FRACTION is not None:
                try:
                    torch.cuda.set_per_process_memory_fraction(config.MAX_PROCESS_VRAM_FRACTION, 0)
                except Exception as e:
                    print(f"[WARNING] Could not set memory fraction: {e}")

        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        
        # Warmup
        if self.device == 'cuda':
            try:
                with torch.inference_mode():
                    dummy_input = torch.zeros((1, 3, 640, 640), device=self.device)
                    if self.use_half: dummy_input = dummy_input.half()
                    for _ in range(2):
                        self.model(dummy_input, verbose=False)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"[WARNING] Warmup failed: {e}")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run pose detection. Uses local model in standalone mode, or sends to server in shared mode.
        """
        if self.shared_mode:
            return self._detect_shared(frame)
        
        self._ensure_model_loaded()
        try:
            with torch.inference_mode():
                results = self.model(frame, conf=0.5, verbose=False, device=self.device, half=self.use_half)
            return self._process_results(results)
        except torch.cuda.OutOfMemoryError:
            print("[ERROR] CUDA OOM Detected! Attempting recovery...")
            torch.cuda.empty_cache()
            try:
                with torch.inference_mode():
                    results = self.model(frame, conf=0.5, verbose=False, device=self.device, half=self.use_half)
                return self._process_results(results)
            except Exception as e:
                print(f"[CRITICAL] Recovery failed: {e}")
                return []
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            return []

    def _detect_shared(self, frame: np.ndarray) -> List[Dict]:
        """Client-side logic for shared inference with single tagged response queue."""
        if self._request_queue is None:
            print("[PoseDetector] WARNING: Shared mode requested but no queues provided. Falling back to standalone.")
            self.shared_mode = False
            return self.detect(frame)

        try:
            if self._shm_buf is not None and self._shm_slot_idx >= 0:
                # Optimized Path: Copy frame to SharedMemory slot
                offset = self._shm_slot_idx * self._shm_slot_size
                if frame.nbytes > self._shm_slot_size:
                    print(f"[PoseDetector] ERROR: Frame too large for SHM slot ({frame.nbytes} > {self._shm_slot_size})")
                    self._request_queue.put((self.client_id, frame))
                else:
                    self._shm_buf[offset:offset+frame.nbytes] = frame.tobytes()
                    metadata = {
                        "shm_slot": self._shm_slot_idx,
                        "shape": frame.shape,
                        "dtype": str(frame.dtype),
                        "nbytes": frame.nbytes
                    }
                    self._request_queue.put((self.client_id, metadata))
            else:
                # Legacy Path: Pass the whole frame through the queue
                self._request_queue.put((self.client_id, frame))

            # Poll shared response queue for a response tagged to this client.
            # Other workers' responses are re-queued to avoid starving them.
            deadline = time.time() + 5.0
            while time.time() < deadline:
                try:
                    tagged = self._response_queue.get(timeout=0.1)
                    recv_id, result = tagged
                    if recv_id == self.client_id:
                        return result
                    else:
                        # Not ours — put it back for the correct worker
                        self._response_queue.put(tagged)
                except Exception:
                    continue
            print(f"[PoseDetector] Shared inference timeout for {self.client_id}")
            return []
        except Exception as e:
            print(f"[PoseDetector] Shared inference error: {e}")
            return []

    def _process_results(self, results) -> List[Dict]:
        """Convert YOLO results to lightweight detection dictionaries."""
        detections = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is not None and result.keypoints is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            boxes_conf = result.boxes.conf.cpu().numpy()
            kpts_xy = result.keypoints.xy.cpu().numpy()
            kpts_conf = result.keypoints.conf.cpu().numpy()

            for i in range(len(boxes_xyxy)):
                kpt_array = np.hstack([
                    kpts_xy[i],
                    kpts_conf[i].reshape(-1, 1)
                ])

                detections.append({
                    "bbox": boxes_xyxy[i],
                    "confidence": float(boxes_conf[i]),
                    "keypoints": kpt_array,
                    "keypoint_names": [
                        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist", "left_hip", "right_hip",
                        "left_knee", "right_knee", "left_ankle", "right_ankle"
                    ]
                })

        self.last_results = detections
        return detections

    def set_queues(self, request_q, response_q, shm_config=None):
        """Set the queues and shared memory used for shared mode."""
        self._request_queue = request_q
        self._response_queue = response_q

        if shm_config and shm_config.get('name'):
            try:
                self._shm = shared_memory.SharedMemory(name=shm_config['name'])
                self._shm_buf = self._shm.buf
                self._shm_slot_size = shm_config['slot_size']
                slot_map = shm_config.get('slot_map', {})
                self._shm_slot_idx = slot_map.get(self.client_id, -1)
                if self._shm_slot_idx >= 0:
                    print(f"[PoseDetector] SharedMemory attached. Slot: {self._shm_slot_idx}")
                else:
                    print(f"[PoseDetector] WARNING: No SharedMemory slot assigned for {self.client_id}")
            except Exception as e:
                print(f"[PoseDetector] Could not attach to SharedMemory: {e}")

    def cleanup(self):
        """Cleanup: detach from SHM."""
        if self._shm:
            self._shm.close()


class _InferenceServer:
    """
    Internal server that manages the GPU model and processes 
    batched requests from multiple workers.
    """
    def __init__(self, model_path=None, device="auto", half=True):
        self.model_path = model_path or config.YOLO_POSE_MODEL
        self.device = device
        self.half = half
        self.running = False
        self.model = None
        self.last_active = time.time()
        self.shm = None
        self.shm_slot_size = 0

    def run(self, request_queue: mp.Queue, response_queue: mp.Queue, shm_name: str = None):
        """Main loop for the inference server with SharedMemory support.
        
        Reads (client_id, payload) from request_queue.
        Writes (client_id, result) to the single shared response_queue.
        Workers filter responses by their own client_id.
        """
        self.running = True
        print("[InferenceServer] Shared GPU Server process started.")

        if shm_name:
            try:
                self.shm = shared_memory.SharedMemory(name=shm_name)
                self.shm_slot_size = (getattr(config, "MAX_SHARED_MEMORY_MB", 1024) * 1024 * 1024) // 100
                print(f"[InferenceServer] SharedMemory attached for frame processing ({shm_name})")
            except Exception as e:
                print(f"[InferenceServer] ERROR: Could not attach to SharedMemory {shm_name}: {e}")

        while self.running:
            requests = []
            try:
                req = request_queue.get(timeout=2.0)
                requests.append(req)
                self.last_active = time.time()
            except:
                if self.model is not None:
                    idle_time = time.time() - self.last_active
                    timeout = getattr(config, "GPU_IDLE_TIMEOUT", 300)
                    if idle_time > timeout:
                        print(f"[InferenceServer] Idle for {idle_time:.0f}s. Unloading model.")
                        self.model = None
                        torch.cuda.empty_cache()
                continue

            if self.model is None:
                self._load_model()

            batch_limit = getattr(config, "BATCH_SIZE_LIMIT", 32)
            wait_ms = getattr(config, "INFERENCE_BATCH_WAIT_MS", 5)
            wait_start = time.perf_counter()

            while len(requests) < batch_limit:
                elapsed = (time.perf_counter() - wait_start) * 1000.0
                remaining = wait_ms - elapsed
                if remaining <= 0: break
                try:
                    req = request_queue.get(timeout=remaining/1000.0)
                    requests.append(req)
                except:
                    break

            if requests:
                client_ids = [r[0] for r in requests]
                payloads = [r[1] for r in requests]
                frames = []

                for payload in payloads:
                    if isinstance(payload, dict) and "shm_slot" in payload:
                        slot_idx = payload["shm_slot"]
                        shape = payload["shape"]
                        dtype = payload["dtype"]
                        nbytes = payload["nbytes"]
                        if self.shm:
                            offset = slot_idx * self.shm_slot_size
                            frame_bytes = self.shm.buf[offset : offset + nbytes]
                            frame = np.frombuffer(frame_bytes, dtype=dtype).reshape(shape).copy()
                            frames.append(frame)
                        else:
                            print(f"[InferenceServer] ERROR: Received SHM payload but SHM not initialized.")
                            frames.append(np.zeros(shape, dtype=dtype))
                    else:
                        frames.append(payload)

                try:
                    with torch.inference_mode():
                        results = self.model(frames, conf=0.5, verbose=False, half=self.half)
                    for i, client_id in enumerate(client_ids):
                        processed = self._process_single_result(results[i])
                        # Tag the reply with client_id so workers can filter their own
                        response_queue.put((client_id, processed))
                except Exception as e:
                    print(f"[InferenceServer] Batch inference error: {e}")
                    for client_id in client_ids:
                        response_queue.put((client_id, []))

    def _load_model(self):
        print(f"[InferenceServer] Initializing model into {self.device}...")
        self.model = YOLO(self.model_path)
        dev = self.device
        if dev == "auto":
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(dev)
        print("[InferenceServer] Model ready.")

    def _process_single_result(self, result) -> List[Dict]:
        detections = []
        if result.boxes is not None and result.keypoints is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            kpts_xy = result.keypoints.xy.cpu().numpy()
            kpts_conf = result.keypoints.conf.cpu().numpy()

            for i in range(len(boxes_xyxy)):
                kpt_array = np.hstack([kpts_xy[i], kpts_conf[i].reshape(-1, 1)])
                detections.append({
                    "bbox": boxes_xyxy[i],
                    "confidence": float(result.boxes.conf[i].cpu().numpy()),
                    "keypoints": kpt_array,
                    "keypoint_names": [
                        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist", "left_hip", "right_hip",
                        "left_knee", "right_knee", "left_ankle", "right_ankle"
                    ]
                })
        return detections
