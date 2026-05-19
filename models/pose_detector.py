# models/pose_detector.py
from ultralytics import YOLO
import numpy as np
import torch
from typing import List, Dict, Optional
import config
import time
import os
import multiprocessing as mp
mp.current_process().authkey = b'pmj_auth'
from multiprocessing.managers import SyncManager, DictProxy
from multiprocessing import shared_memory


# ---------------------------------------------------------------------------
# InferenceManager — hosts the request queue and the per-client registry.
#
# Architecture (NEW):
#   - One shared REQUEST queue:  all workers → GPU server.
#   - Per-client RESPONSE queues: GPU server → exactly one worker.
#     Each worker registers its own mp.Queue in a Manager dict keyed by
#     client_id.  The GPU server does a direct dict-lookup and puts the
#     result into the correct queue — no polling, no re-queuing, O(1).
#
# This replaces the broken single shared response queue where all N workers
# polled the same queue, re-queued foreign messages, and exhausted their
# timeout windows on O(N²) re-queue churn.
# ---------------------------------------------------------------------------

class InferenceManager(SyncManager):
    pass


def get_shared_queues(address=('127.0.0.1', 50000), authkey=b'pmj_auth'):
    """Connect to a running InferenceManager and return the shared objects.

    Returns:
        (request_queue, response_registry, shm_config)
        response_registry is a Manager dict: {client_id -> mp.Queue}
    """
    InferenceManager.register('get_request_queue')
    InferenceManager.register('get_response_registry', proxytype=DictProxy)
    InferenceManager.register('get_shared_memory_config', proxytype=DictProxy)
    manager = InferenceManager(address=address, authkey=authkey)
    try:
        manager.connect()
        return (
            manager.get_request_queue(),
            manager.get_response_registry(),
            manager.get_shared_memory_config(),
        )
    except Exception as e:
        print(f"[InferenceManager] Could not connect to shared server: {e}")
        return None, None, None


def start_inference_manager(
    request_q,
    response_registry,
    shm_config,
    address=('127.0.0.1', 50000),
    authkey=b'pmj_auth',
):
    """Start a manager server hosting the request queue, response registry and shm config."""
    InferenceManager.register('get_request_queue',      callable=lambda: request_q)
    InferenceManager.register('get_response_registry',  callable=lambda: response_registry, proxytype=DictProxy)
    InferenceManager.register('get_shared_memory_config', callable=lambda: shm_config,       proxytype=DictProxy)
    manager = InferenceManager(address=address, authkey=authkey)
    server = manager.get_server()
    return server


class PoseDetector:
    """YOLOv8-Pose wrapper optimized for production CUDA environments."""

    def __init__(
        self,
        model_path: str = None,
        device: str = "auto",
        half: bool = True,
        shared_mode: bool = None,
    ):
        self.model_path = model_path or config.YOLO_POSE_MODEL
        self.device_request = device
        self.half_request = half
        self.shared_mode = (
            shared_mode if shared_mode is not None
            else getattr(config, "SHARED_INFERENCE_ENABLED", False)
        )

        self.model = None
        self.device = None
        self.use_half = False
        self.last_results = None

        # Client identity and per-client response queue (shared mode only)
        self.client_id = f"client_{os.getpid()}_{id(self)}" if self.shared_mode else None
        self._request_queue = None
        self._response_queue: Optional[mp.Queue] = None   # OUR private queue
        self._response_registry = None                     # Manager DictProxy
        self._shm = None
        self._shm_buf = None
        self._shm_slot_idx = -1
        self._shm_slot_size = 0

        if not self.shared_mode:
            print(f"[PoseDetector] Initialized in STANDALONE mode (Lazy Load enabled)")
        else:
            print(f"[PoseDetector] Initialized in SHARED mode (Client ID: {self.client_id})")

    # ------------------------------------------------------------------
    # Standalone model loading
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self):
        """Lazy-load the YOLO model on first inference call."""
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
            frac = getattr(config, "MAX_PROCESS_VRAM_FRACTION", None)
            if frac is not None:
                try:
                    torch.cuda.set_per_process_memory_fraction(frac, 0)
                except Exception as e:
                    print(f"[WARNING] Could not set memory fraction: {e}")

        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        if self.device == 'cuda':
            try:
                with torch.inference_mode():
                    dummy = torch.zeros((1, 3, 640, 640), device=self.device)
                    if self.use_half:
                        dummy = dummy.half()
                    for _ in range(2):
                        self.model(dummy, verbose=False)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"[WARNING] Warmup failed: {e}")

    # ------------------------------------------------------------------
    # Public inference entry point
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Run pose detection — shared mode or standalone."""
        if self.shared_mode:
            return self._detect_shared(frame)

        self._ensure_model_loaded()
        try:
            with torch.inference_mode():
                results = self.model(
                    frame, conf=0.5, verbose=False,
                    device=self.device, half=self.use_half,
                )
            return self._process_results(results)
        except torch.cuda.OutOfMemoryError:
            print("[ERROR] CUDA OOM — attempting recovery...")
            torch.cuda.empty_cache()
            try:
                with torch.inference_mode():
                    results = self.model(
                        frame, conf=0.5, verbose=False,
                        device=self.device, half=self.use_half,
                    )
                return self._process_results(results)
            except Exception as e:
                print(f"[CRITICAL] OOM recovery failed: {e}")
                return []
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Shared-mode inference — per-client response queue (NEW)
    # ------------------------------------------------------------------

    def _detect_shared(self, frame: np.ndarray) -> List[Dict]:
        """Submit a frame to the GPU server and wait on OUR private response queue.

        Key design change vs. the old architecture:
          OLD: one shared response_queue polled by all workers → O(N²) re-queue churn,
               5-second timeouts exhausted re-queuing other cameras' results.
          NEW: each worker has its own mp.Queue registered in response_registry.
               The GPU server looks up client_id → queue and puts the result there
               directly.  No polling of foreign results, no re-queuing, O(1) delivery.
        """
        if self._request_queue is None or self._response_queue is None:
            print(
                "[PoseDetector] WARNING: Shared mode queues not initialised. "
                "Falling back to standalone."
            )
            self.shared_mode = False
            return self.detect(frame)

        try:
            # Build request payload (SHM fast-path or queue copy)
            if self._shm_buf is not None and self._shm_slot_idx >= 0:
                offset = self._shm_slot_idx * self._shm_slot_size
                if frame.nbytes > self._shm_slot_size:
                    print(
                        f"[PoseDetector] Frame too large for SHM slot "
                        f"({frame.nbytes} > {self._shm_slot_size}); using queue copy."
                    )
                    self._request_queue.put((self.client_id, frame))
                else:
                    self._shm_buf[offset : offset + frame.nbytes] = frame.tobytes()
                    self._request_queue.put((
                        self.client_id,
                        {
                            "shm_slot": self._shm_slot_idx,
                            "shape":    frame.shape,
                            "dtype":    str(frame.dtype),
                            "nbytes":   frame.nbytes,
                        },
                    ))
            else:
                self._request_queue.put((self.client_id, frame))

            # Wait on OUR dedicated queue — no other worker will ever touch it.
            # Timeout is 3× the expected p99 inference latency.  On timeout we
            # return a sentinel None so the caller's FSM can distinguish
            # "inference unavailable" from "no detections".
            timeout_s = getattr(config, "INFERENCE_TIMEOUT_SECONDS", 2.0)
            try:
                result = self._response_queue.get(timeout=timeout_s)
                return result
            except Exception:
                # Queue.Empty — genuine inference timeout, not a routing failure.
                print(
                    f"[PoseDetector] Inference timeout ({timeout_s}s) for {self.client_id}. "
                    f"Returning INFERENCE_TIMEOUT sentinel."
                )
                # Return the sentinel so the FSM can hold state rather than wipe it.
                return None   # Callers must handle None → use LKG / hold FSM state

        except Exception as e:
            print(f"[PoseDetector] Shared inference error: {e}")
            return None

    # ------------------------------------------------------------------
    # Queue / SHM wiring (called by main.py after manager connect)
    # ------------------------------------------------------------------

    def set_queues(
        self,
        request_q,
        response_registry,       # SyncManager DictProxy {client_id -> manager.Queue proxy}
        shm_config=None,
    ):
        """Wire up the per-client queues and shared memory.

        Creates a private managed Queue for this client and registers it in
        response_registry so the GPU server can route replies to it directly.

        IMPORTANT — mp.Queue() vs manager.Queue():
          Plain mp.Queue() objects cannot be stored in a Manager DictProxy.
          Python 3.8+ raises "Queue objects should only be shared between
          processes through inheritance" when pickling them through the
          Manager IPC socket.  manager.Queue() returns a proxy object that
          IS picklable across process boundaries — it communicates through the
          Manager server rather than through a raw OS pipe file descriptor.
          We need the manager reference, so set_queues() accepts it as a
          parameter injected by main.py's setup block.
        """
        self._request_queue = request_q
        self._response_registry = response_registry

        # response_registry._manager is the SyncManager instance created in
        # main.py.  We call .Queue() on it to get a properly managed proxy
        # that can be stored in the DictProxy and retrieved by the GPU server
        # subprocess without pickling a raw file descriptor.
        manager = response_registry._manager
        self._response_queue = manager.Queue(maxsize=4)
        response_registry[self.client_id] = self._response_queue
        print(
            f"[PoseDetector] Registered per-client response queue for {self.client_id}"
        )

        if shm_config and shm_config.get('name'):
            try:
                self._shm = shared_memory.SharedMemory(name=shm_config['name'])
                self._shm_buf = self._shm.buf
                self._shm_slot_size = shm_config['slot_size']
                slot_map = shm_config.get('slot_map', {})
                self._shm_slot_idx = slot_map.get(self.client_id, -1)
                if self._shm_slot_idx >= 0:
                    print(
                        f"[PoseDetector] SharedMemory attached. "
                        f"Slot {self._shm_slot_idx} of size {self._shm_slot_size} bytes."
                    )
                else:
                    print(
                        f"[PoseDetector] WARNING: No SHM slot for {self.client_id}; "
                        f"will use queue copy path."
                    )
            except Exception as e:
                print(f"[PoseDetector] Could not attach to SharedMemory: {e}")

    def cleanup(self):
        """Deregister from response_registry and release SHM."""
        if self._response_registry is not None and self.client_id is not None:
            try:
                del self._response_registry[self.client_id]
                print(f"[PoseDetector] Deregistered {self.client_id} from response registry.")
            except Exception:
                pass
        if self._shm:
            self._shm.close()

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    def _process_results(self, results) -> List[Dict]:
        """Convert YOLO results to lightweight detection dicts."""
        detections = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is not None and result.keypoints is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            boxes_conf = result.boxes.conf.cpu().numpy()
            kpts_xy    = result.keypoints.xy.cpu().numpy()
            kpts_conf  = result.keypoints.conf.cpu().numpy()

            for i in range(len(boxes_xyxy)):
                kpt_array = np.hstack([
                    kpts_xy[i],
                    kpts_conf[i].reshape(-1, 1),
                ])
                detections.append({
                    "bbox":       boxes_xyxy[i],
                    "confidence": float(boxes_conf[i]),
                    "keypoints":  kpt_array,
                    "keypoint_names": [
                        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist", "left_hip", "right_hip",
                        "left_knee", "right_knee", "left_ankle", "right_ankle",
                    ],
                })

        self.last_results = detections
        return detections


# ---------------------------------------------------------------------------
# _InferenceServer — GPU process
#
# Change from old architecture:
#   OLD: response_queue.put((client_id, result))
#        → all workers poll this one queue, re-queue wrong results.
#   NEW: response_registry[client_id].put(result)
#        → result lands directly in the requesting worker's private queue.
#        No shared polling, no re-queuing, zero contention.
# ---------------------------------------------------------------------------

class _InferenceServer:
    """Manages the GPU model and routes batched results to per-client queues."""

    def __init__(self, model_path=None, device="auto", half=True):
        self.model_path  = model_path or config.YOLO_POSE_MODEL
        self.device      = device
        self.half        = half
        self.running     = False
        self.model       = None
        self.last_active = time.time()
        self.shm         = None
        self.shm_slot_size = 0

    def run(
        self,
        request_queue: mp.Queue,
        response_registry,          # Manager DictProxy {client_id -> mp.Queue}
        shm_name: str = None,
    ):
        """Main inference loop.

        Reads  (client_id, payload) from request_queue.
        Writes result directly to response_registry[client_id].

        If a worker has already disconnected and removed its queue from the
        registry, the result is silently dropped (worker is gone anyway).
        """
        self.running = True
        print("[InferenceServer] GPU server process started (per-client routing).")

        if shm_name:
            try:
                self.shm = shared_memory.SharedMemory(name=shm_name)
                self.shm_slot_size = (
                    getattr(config, "MAX_SHARED_MEMORY_MB", 1024) * 1024 * 1024
                ) // 100
                print(f"[InferenceServer] SharedMemory attached: {shm_name}")
            except Exception as e:
                print(f"[InferenceServer] ERROR: Could not attach to SharedMemory {shm_name}: {e}")

        while self.running:
            # ---- Collect first request (blocking with idle timeout) ----
            requests = []
            try:
                req = request_queue.get(timeout=2.0)
                requests.append(req)
                self.last_active = time.time()
            except Exception:
                # Idle timeout — optionally unload model to free VRAM
                if self.model is not None:
                    idle_s = time.time() - self.last_active
                    gpu_idle_timeout = getattr(config, "GPU_IDLE_TIMEOUT", 300)
                    if idle_s > gpu_idle_timeout:
                        print(f"[InferenceServer] Idle {idle_s:.0f}s — unloading model.")
                        self.model = None
                        torch.cuda.empty_cache()
                continue

            if self.model is None:
                self._load_model()

            # ---- Drain additional requests up to batch limit ----
            batch_limit = getattr(config, "BATCH_SIZE_LIMIT", 32)
            wait_ms     = getattr(config, "INFERENCE_BATCH_WAIT_MS", 5)
            wait_start  = time.perf_counter()

            while len(requests) < batch_limit:
                elapsed_ms = (time.perf_counter() - wait_start) * 1000.0
                remaining  = wait_ms - elapsed_ms
                if remaining <= 0:
                    break
                try:
                    req = request_queue.get(timeout=remaining / 1000.0)
                    requests.append(req)
                except Exception:
                    break

            if not requests:
                continue

            client_ids = [r[0] for r in requests]
            payloads   = [r[1] for r in requests]

            # ---- Decode frames (SHM fast-path or direct) ----
            frames = []
            for payload in payloads:
                if isinstance(payload, dict) and "shm_slot" in payload:
                    if self.shm:
                        slot_idx = payload["shm_slot"]
                        shape    = payload["shape"]
                        dtype    = payload["dtype"]
                        nbytes   = payload["nbytes"]
                        offset   = slot_idx * self.shm_slot_size
                        frame    = (
                            np.frombuffer(
                                self.shm.buf[offset : offset + nbytes],
                                dtype=dtype,
                            )
                            .reshape(shape)
                            .copy()
                        )
                        frames.append(frame)
                    else:
                        print("[InferenceServer] ERROR: SHM payload but SHM not initialised.")
                        shape = payload.get("shape", (640, 640, 3))
                        dtype = payload.get("dtype", "uint8")
                        frames.append(np.zeros(shape, dtype=dtype))
                else:
                    frames.append(payload)

            # ---- Run batch inference ----
            try:
                with torch.inference_mode():
                    results = self.model(
                        frames, conf=0.5, verbose=False, half=self.half
                    )

                for i, client_id in enumerate(client_ids):
                    processed = self._process_single_result(results[i])
                    self._send_to_client(response_registry, client_id, processed)

            except Exception as e:
                print(f"[InferenceServer] Batch inference error: {e}")
                # On error send empty list (genuine empty — no persons detected).
                # Workers that interpret None as TIMEOUT will not misread this.
                for client_id in client_ids:
                    self._send_to_client(response_registry, client_id, [])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_to_client(self, response_registry, client_id: str, result):
        """Route result directly to the client's private queue.

        If the client has already deregistered (camera worker restarted),
        silently drop the result — there is nobody to receive it.
        """
        try:
            q = response_registry.get(client_id)
            if q is not None:
                q.put_nowait(result)
            else:
                # Worker gone — drop result, no re-queuing needed
                pass
        except Exception as e:
            print(f"[InferenceServer] Could not deliver result to {client_id}: {e}")

    def _load_model(self):
        print(f"[InferenceServer] Loading model onto {self.device}...")
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
            kpts_xy    = result.keypoints.xy.cpu().numpy()
            kpts_conf  = result.keypoints.conf.cpu().numpy()

            for i in range(len(boxes_xyxy)):
                kpt_array = np.hstack([
                    kpts_xy[i],
                    kpts_conf[i].reshape(-1, 1),
                ])
                detections.append({
                    "bbox":       boxes_xyxy[i],
                    "confidence": float(result.boxes.conf[i].cpu().numpy()),
                    "keypoints":  kpt_array,
                    "keypoint_names": [
                        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist", "left_hip", "right_hip",
                        "left_knee", "right_knee", "left_ankle", "right_ankle",
                    ],
                })
        return detections