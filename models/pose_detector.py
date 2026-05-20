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
        (scan_req_q, priority_req_q, response_registry, priority_registry, shm_config)
        response_registry: Manager dict {client_id -> mp.Queue}  (routing)
        priority_registry: Manager dict {client_id -> str}        (observability)
    """
    InferenceManager.register('get_scan_queue')
    InferenceManager.register('get_priority_queue')
    InferenceManager.register('get_response_registry', proxytype=DictProxy)
    InferenceManager.register('get_priority_registry', proxytype=DictProxy)
    InferenceManager.register('get_shared_memory_config', proxytype=DictProxy)
    manager = InferenceManager(address=address, authkey=authkey)
    try:
        manager.connect()
        return (
            manager.get_scan_queue(),
            manager.get_priority_queue(),
            manager.get_response_registry(),
            manager.get_priority_registry(),
            manager.get_shared_memory_config(),
        )
    except Exception as e:
        print(f"[InferenceManager] Could not connect to shared server: {e}")
        return None, None, None, None, None


def start_inference_manager(
    scan_req_q,
    priority_req_q,
    response_registry,
    priority_registry,
    shm_config,
    address=('127.0.0.1', 50000),
    authkey=b'pmj_auth',
):
    """Start a manager server hosting both request queues, response registry,
    priority registry and shm config."""
    InferenceManager.register('get_scan_queue',          callable=lambda: scan_req_q)
    InferenceManager.register('get_priority_queue',      callable=lambda: priority_req_q)
    InferenceManager.register('get_response_registry',   callable=lambda: response_registry,  proxytype=DictProxy)
    InferenceManager.register('get_priority_registry',   callable=lambda: priority_registry,  proxytype=DictProxy)
    InferenceManager.register('get_shared_memory_config',callable=lambda: shm_config,          proxytype=DictProxy)
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
        self._priority_registry  = None                     # observability dict
        self._shm = None
        self._shm_buf = None
        self._shm_slot_idx = -1
        self._shm_slot_size = 0

        # Dual request queues — workers self-route based on FSM state
        self._scanning_request_queue = None   # Low-priority: idle/background streams
        self._priority_request_queue = None   # High-priority: active unlocker streams

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

    def detect(self, frame: np.ndarray, fsm_is_active: bool = False) -> List[Dict]:
        """Run pose detection — shared mode or standalone.

        fsm_is_active: True when candidate_a, candidate_b, id_a, or id_b is set
        in the state machine session (i.e., an unlocker interaction is in progress).
        In shared mode this routes the request to the dedicated priority queue so
        the active stream gets faster GPU turnaround than idle scanning streams.
        """
        if self.shared_mode:
            return self._detect_shared(frame, fsm_is_active=fsm_is_active)

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

    def _detect_shared(self, frame: np.ndarray, fsm_is_active: bool = False) -> List[Dict]:
        """Submit a frame to the GPU server and wait on OUR private response queue.

        Self-routing design:
          idle stream  (fsm_is_active=False) → scanning request queue
                                                (batch=16, lower priority)
          active stream (fsm_is_active=True)  → priority request queue
                                                (batch=4, fast GPU turnaround)

        Workers decide routing locally from their own FSM state — no cross-process
        calls needed. The supervisor cannot reach into child process objects.
        """
        # Pick the correct request queue based on current FSM activity.
        # Fall back to scanning queue if priority queue isn't wired yet.
        req_q = (
            self._priority_request_queue
            if (fsm_is_active and self._priority_request_queue is not None)
            else self._scanning_request_queue
        )

        if req_q is None or self._response_queue is None:
            print(
                "[PoseDetector] WARNING: Shared mode queues not initialised. "
                "Falling back to standalone."
            )
            self.shared_mode = False
            return self.detect(frame)

        # Observability: write current tier to priority_registry (best-effort, non-fatal)
        if self._priority_registry is not None:
            try:
                self._priority_registry[self.client_id] = (
                    "ACTIVE" if fsm_is_active else "SCANNING"
                )
            except Exception:
                pass

        try:
            # Build request payload (SHM fast-path or queue copy)
            if self._shm_buf is not None and self._shm_slot_idx >= 0:
                offset = self._shm_slot_idx * self._shm_slot_size
                if frame.nbytes > self._shm_slot_size:
                    print(
                        f"[PoseDetector] Frame too large for SHM slot "
                        f"({frame.nbytes} > {self._shm_slot_size}); using queue copy."
                    )
                    req_q.put((self.client_id, frame))
                else:
                    self._shm_buf[offset : offset + frame.nbytes] = frame.tobytes()
                    req_q.put((
                        self.client_id,
                        {
                            "shm_slot": self._shm_slot_idx,
                            "shape":    frame.shape,
                            "dtype":    str(frame.dtype),
                            "nbytes":   frame.nbytes,
                        },
                    ))
            else:
                req_q.put((self.client_id, frame))

            # Tier-appropriate timeout:
            #   Priority streams get a tighter deadline — measured p99 * 2.
            #   Scanning streams keep the safe 2.0s default.
            # Both default to INFERENCE_TIMEOUT_SECONDS until GPU latency is confirmed.
            timeout_s = (
                getattr(config, "PRIORITY_INFERENCE_TIMEOUT_SECONDS",
                        getattr(config, "INFERENCE_TIMEOUT_SECONDS", 2.0))
                if fsm_is_active
                else getattr(config, "INFERENCE_TIMEOUT_SECONDS", 2.0)
            )
            try:
                result = self._response_queue.get(timeout=timeout_s)
                return result
            except Exception:
                tier = "PRIORITY" if fsm_is_active else "SCANNING"
                print(
                    f"[PoseDetector] {tier} inference timeout ({timeout_s}s) "
                    f"for {self.client_id}. Returning INFERENCE_TIMEOUT sentinel."
                )
                return None   # Callers handle None → use LKG / hold FSM state

        except Exception as e:
            print(f"[PoseDetector] Shared inference error: {e}")
            return None

    # ------------------------------------------------------------------
    # Queue / SHM wiring (called by main.py after manager connect)
    # ------------------------------------------------------------------

    def set_queues(
        self,
        scan_req_q,
        priority_req_q,
        response_registry,       # SyncManager DictProxy {client_id -> manager.Queue proxy}
        priority_registry=None,  # SyncManager DictProxy {client_id -> "SCANNING"|"ACTIVE"}
        shm_config=None,
    ):
        """Wire up the per-client queues and shared memory.

        Creates a private managed Queue for this client and registers it in
        response_registry so the GPU server can route replies to it directly.

        Dual-queue routing:
          scan_req_q     — submitted when fsm_is_active=False (idle/background streams)
          priority_req_q — submitted when fsm_is_active=True  (active unlocker streams)
        Workers self-route each frame based on their own local FSM state.

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
        self._scanning_request_queue = scan_req_q
        self._priority_request_queue = priority_req_q
        self._response_registry      = response_registry
        self._priority_registry      = priority_registry

        # Keep _request_queue as alias of scanning queue for any legacy code paths
        self._request_queue = scan_req_q

        # response_registry._manager is the SyncManager instance created in
        # main.py.  We call .Queue() on it to get a properly managed proxy
        # that can be stored in the DictProxy and retrieved by the GPU server
        # subprocess without pickling a raw file descriptor.
        manager = response_registry._manager
        self._response_queue = manager.Queue(maxsize=4)
        response_registry[self.client_id] = self._response_queue
        print(
            f"[PoseDetector] Registered per-client response queue for {self.client_id} "
            f"(scan_q={'OK' if scan_req_q else 'MISSING'}, "
            f"priority_q={'OK' if priority_req_q else 'MISSING'})"
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
        is_priority: bool = False,  # True = priority server (small batch, fast)
    ):
        """Main inference loop.

        Reads  (client_id, payload) from request_queue.
        Writes result directly to response_registry[client_id].

        If a worker has already disconnected and removed its queue from the
        registry, the result is silently dropped (worker is gone anyway).
        """
        self.running    = True
        self.is_priority = is_priority
        tier_label = "PRIORITY" if is_priority else "SCANNING"
        print(f"[InferenceServer] GPU server process started ({tier_label} tier, per-client routing).")

        if shm_name:
            try:
                self.shm = shared_memory.SharedMemory(name=shm_name)
                self.shm_slot_size = (
                    getattr(config, "MAX_SHARED_MEMORY_MB", 1024) * 1024 * 1024
                ) // 100
                print(f"[InferenceServer] SharedMemory attached: {shm_name}")
            except Exception as e:
                print(f"[InferenceServer] ERROR: Could not attach to SharedMemory {shm_name}: {e}")

        # Pre-warm model before first request if configured.
        # Eliminates cold-start latency spike at window open.
        if getattr(config, "GPU_PRELOAD_ON_START", False):
            self._load_model()

        batch_limit = (
            getattr(config, "PRIORITY_BATCH_SIZE_LIMIT", 4)
            if is_priority
            else getattr(config, "BATCH_SIZE_LIMIT", 16)
        )
        print(f"[InferenceServer] {tier_label} tier ready. Batch limit: {batch_limit}.")

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
                        tier_label = "PRIORITY" if self.is_priority else "SCANNING"
                        print(f"[InferenceServer] {tier_label} idle {idle_s:.0f}s — unloading model.")
                        self.model = None
                        torch.cuda.empty_cache()
                continue

            if self.model is None:
                self._load_model()

            # ---- Drain additional requests up to batch limit ----
            # batch_limit set per-tier in run() based on is_priority flag
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