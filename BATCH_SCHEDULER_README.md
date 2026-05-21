# Zero-Latency Batch Scheduler Guide

## Overview

The **Batch Scheduler** allows you to add unlimited streams to your GPU without increasing per-frame latency. It works by:

1. **Fixed Cadence**: All streams feed frames into a shared GPU worker that runs on a fixed schedule (e.g., every 33ms for 30 FPS).
2. **Batching**: Multiple streams' frames are inferred in a single GPU pass, so throughput scales with batch size.
3. **Per-Stream Queues**: Results are routed back to each stream independently, no global blocking.
4. **Predictable Memory**: GPU memory is pre-allocated at startup, avoiding allocation spikes mid-run.

## Key Benefit

**No latency increase when adding streams.** If your GPU can handle `N` frames/second, the batch scheduler serves `N` frames from any number of streams with constant p99 latency.

---

## Quick Start

### 1. Enable Batch Scheduler in `config.py`

```python
# ============ ZERO-LATENCY BATCH SCHEDULER ============
BATCH_SCHEDULER_ENABLED = True          # Enable batch coordinator

BATCH_SIZE = 4                          # Frames per GPU pass (default: 4)
BATCH_INFERENCE_CADENCE_MS = 33.0      # 30 FPS (1000/30 = 33.3ms)
BATCH_GPU_PREALLOCATE_MB = 1024.0      # Pre-alloc 1GB on startup

BATCH_INPUT_QUEUE_SIZE = 5              # Frames buffered per stream
BATCH_OUTPUT_QUEUE_SIZE = 10            # Results buffered per stream
```

### 2. Run Multiple Streams

```bash
python main.py \
  --stream-video 4=GF-5-25-12-M.mp4 \
  --stream-video 5=FF-3-21-13-M.mp4 \
  --stream-video 6=GF-6-18-18-M.mp4 \
  --stream-video 7=GF-10-29-18-M.mp4 \
  --device cuda \
  --process-every 1 \
  --test-window morning
```

All 4 streams share **one GPU worker** running at 30 FPS. No stream sees increased latency as you add more.

---

## Tuning for Your GPU

### How Much Does Batch Size Matter?

- **BATCH_SIZE = 1**: Like standalone mode, but single GPU worker. Lower throughput.
- **BATCH_SIZE = 2-4**: Sweet spot for most GPUs. Good throughput, low latency.
- **BATCH_SIZE = 8+**: Higher throughput, but requires more VRAM and slightly higher latency.

**Formula**: Inference time per batch = (constant YOLO overhead) + (frames × per-frame compute).

With batching, the overhead is amortized: 4 frames inferred ~≈ 2× time of 1 frame (not 4×).

### Cadence and FPS

- Set `BATCH_INFERENCE_CADENCE_MS = 1000 / desired_FPS`.
- Example: `BATCH_INFERENCE_CADENCE_MS = 50` → 20 FPS (1000/50).
- Example: `BATCH_INFERENCE_CADENCE_MS = 33.3` → 30 FPS (1000/33.3).

The scheduler **always runs the batch at this cadence**, so adding streams doesn't shift the schedule — it just makes each batch "fuller."

### GPU Pre-Allocation

- `BATCH_GPU_PREALLOCATE_MB = 1024` reserves 1GB at startup.
- Helps avoid CUDA allocation delays mid-run.
- Set based on your GPU VRAM and BATCH_SIZE:
  - **RTXA5000** (24GB): `2048` MB is safe.
  - **RTX3090** (24GB): `1536` MB is safe.
  - **RTX4060** (8GB): `512` MB is safe.

### Input/Output Queue Sizes

- `BATCH_INPUT_QUEUE_SIZE = 5`: If a stream's input queue fills, oldest frame is dropped (not queued).
- `BATCH_OUTPUT_QUEUE_SIZE = 10`: If results pile up faster than the stream consumes them, oldest result is dropped.
- Tune these based on stream processing speed:
  - If `process_every = 3`, you need less buffering.
  - If `process_every = 1`, you may need more.

---

## Frame Drop Behavior

The batch scheduler tracks **per-stream frame drops**:

```
[METRICS] Stream 4 | Frames: 1250 | Dropped: 2 | Queue sizes: in=1, out=0
```

- **Dropped input frames**: Stream's input queue was full (GPU inference faster than frame arrival).
  - *This is OK*: means your GPU can keep up.
- **Dropped output frames**: Inference results piling up (stream processing slower than inference).
  - *This is OK* for audit windows (you're sampling results anyway).
  - *Monitor if too high*: stream may be too slow.

---

## Example Configurations

### Light Load (1-2 streams, low-end GPU)

```python
BATCH_SCHEDULER_ENABLED = True
BATCH_SIZE = 2
BATCH_INFERENCE_CADENCE_MS = 50.0      # 20 FPS
BATCH_GPU_PREALLOCATE_MB = 512.0
```

### Medium Load (4-6 streams, modern GPU)

```python
BATCH_SCHEDULER_ENABLED = True
BATCH_SIZE = 4
BATCH_INFERENCE_CADENCE_MS = 33.0      # 30 FPS
BATCH_GPU_PREALLOCATE_MB = 1024.0
```

### Heavy Load (8+ streams, high-end GPU)

```python
BATCH_SCHEDULER_ENABLED = True
BATCH_SIZE = 8
BATCH_INFERENCE_CADENCE_MS = 25.0      # 40 FPS
BATCH_GPU_PREALLOCATE_MB = 2048.0
```

---

## Comparing Batch vs. Standalone

| Aspect | Standalone | Batch Scheduler |
|--------|-----------|---|
| **Latency with N streams** | Increases with N | Fixed (constant) |
| **GPU Memory** | Unpredictable spikes | Pre-allocated, smooth |
| **Throughput (frames/sec)** | Limited by GPU contention | Scales with batch size |
| **Frame Drop Rate** | Per-stream variable | Predictable per queue size |
| **Tuning** | Per-stream device, per-stream process_every | Single batch config |

---

## How to Monitor

Enable debug logging:

```bash
python main.py \
  --stream-video 4=video.mp4 \
  --stream-video 5=video2.mp4 \
  --device cuda \
  --debug
```

Look for:
- `[GPUBatchCoordinator] frames_processed=N batches_run=M inference_time_ms=X`
- Per-stream frame drop counts in `[METRICS]` lines.

---

## Fallback to Standalone If Needed

If you encounter issues with the batch scheduler:

```python
BATCH_SCHEDULER_ENABLED = False
```

This reverts to the standalone GPU mode (one process per stream, direct GPU inference).

---

## FAQ

**Q: Will batch scheduler work with `--show` and `--debug`?**  
A: Yes. `--show` renders windows for each stream independently, batch scheduler just handles GPU inference.

**Q: Can I change batch size mid-run?**  
A: No. The batch size is set at coordinator startup. Restart the process to change it.

**Q: What if my GPU runs out of memory?**  
A: Reduce `BATCH_SIZE` or `BATCH_GPU_PREALLOCATE_MB`. The coordinator will print a warning and skip that batch.

**Q: Does batch scheduler work with `--shared-inference`?**  
A: No. Batch scheduler **replaces** shared inference. Set `BATCH_SCHEDULER_ENABLED=True` and `SHARED_INFERENCE_ENABLED=False`.

---

## Next Steps

1. Start with `BATCH_SCHEDULER_ENABLED = True` and `BATCH_SIZE = 4`.
2. Add streams one at a time and monitor `[METRICS]` for frame drops and inference time.
3. If inference time > cadence (e.g., 40ms for 33ms cadence), reduce `BATCH_SIZE`.
4. If GPU is underutilized (low inference time), try increasing `BATCH_SIZE` or `--process-every 1`.
