> For the complete project and a new laptop installation, start with the [root README](../README.md). The working web dashboard is launched from the project root with Start AURA.cmd. The engine documentation below records the earlier enhancement milestones.

# AURA Video Enhancement Engine — Phase 0

## Benchmark & Streaming Foundation

AURA (Autonomous Video Enhancement Engine) Phase 0 provides a high-throughput, low-latency video ingestion, processing, and performance benchmarking foundation designed for real-time First Person View (FPV) video streams.

---

## Key Features (Phase 0 Scope)

* **Low-Latency Streaming Capture**: Non-blocking threaded video reader with bounded frame buffer queue (`maxsize=2` by default) to prevent queue accumulation and ensure fresh frame delivery.
* **Microsecond Frame Precision**: Exact per-frame timestamps recorded using high-resolution performance counters (`time.perf_counter()`).
* **Modular Pipeline Architecture**: Decoupled `BaseProcessor` class allowing seamless integration of future enhancement models without altering capture, metric, display, or recording modules.
* **Identity Baseline Processor**: Standardized identity pass-through (`input frame -> output frame`) preserving shape, color channels, and `uint8` array data for baseline latency measurement.
* **Comprehensive Metrics Tracker**: Measures input FPS, output processing FPS, average frame time, median frame time, P95 frame time, min/max frame latencies, and dropped frames.
* **Fail-Safe Hardware Stats**: Safe extraction of CPU %, RAM %, and NVIDIA GPU utilization / VRAM (via `psutil` / `pynvml`) with safe `"N/A"` fallbacks when unavailable.
* **Live HUD Overlay Display**: Real-time OpenCV display window rendering live FPS, frame processing latencies, resolution, and dropped frame counts.
* **Robust Video Recording**: Output stream writer with automatic codec fallback (`mp4v` -> `avc1` -> `XVID` -> `MJPG`).
* **Automated Benchmark Runner**: CLI benchmark mode producing both human-readable text reports and machine-readable JSON artifacts.

---

## Architecture

```text
Input Source (Camera / File / RTSP)
         ↓
   Capture Layer (Threaded + Bounded Buffer)
         ↓
  Frame Microsecond Timestamp
         ↓
  Processing Layer (Identity Baseline)
         ↓
   Display Layer (Live HUD Overlay)
         ↓
   Video Recorder (Optional Disk Writer)
         ↓
   Benchmark Engine & JSON/TXT Logger
```

### Low-Latency Buffer Design Rationale
For real-time FPV video applications, frame freshness is critical. If frame processing falls behind capture speed, an unbounded queue builds latency, delivering stale historical frames to the pilot or downstream models.

AURA implements a bounded buffer queue. When the processing consumer falls behind, the capture thread automatically drops stale frames, increments the `dropped_frames` metric, and guarantees that the consumer always receives the latest available frame.

---

## Installation

### Prerequisites
* Python 3.9+
* OpenCV & NumPy

### Setup
```bash
# Clone or navigate to the workspace directory
cd "c:/Users/azhar/CODES/MAJOR PROJECT/feeed enhancement"

# Install core dependencies
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Generating a Synthetic Test Video
Generate a synthetic 60 FPS 1280x720 video for testing without needing a physical camera:
```bash
python scripts/generate_sample_video.py
```

### 2. Running Live Mode (Camera or File)
```bash
# Webcam (index 0)
python main.py --source 0

# Video file input
python main.py --source samples/test_synthetic.mp4

# Video file with recording enabled
python main.py --source samples/test_synthetic.mp4 --record
```

### 3. Headless Benchmark Mode
Run a 10-second benchmark run without displaying a window:
```bash
python scripts/benchmark.py --source samples/test_synthetic.mp4 --benchmark-duration 10 --no-display
```

or using `python -m app.pipeline`:
```bash
python -m app.pipeline --source samples/test_synthetic.mp4 --benchmark-duration 10 --no-display
```

---

## CLI Parameters

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source` | `str/int` | `0` | Camera index (e.g. `0`) or video file path (e.g. `samples/test_synthetic.mp4`) |
| `--width` | `int` | `0` | Requested capture width (0 for native) |
| `--height` | `int` | `0` | Requested capture height (0 for native) |
| `--fps` | `float` | `0.0` | Requested capture FPS (0 for native) |
| `--benchmark-duration` | `float` | `0.0` | Execution limit in seconds for benchmark mode (0 for continuous) |
| `--max-frames` | `int` | `0` | Limit total frames processed before exiting (0 for unlimited) |
| `--output-dir` | `str` | `outputs` | Directory path to store JSON/TXT benchmark reports and video recordings |
| `--record` | `flag` | `False` | Enables recording output stream to disk |
| `--no-display` | `flag` | `False` | Disables live GUI display window (headless mode) |
| `--queue-size` | `int` | `2` | Bounded capture buffer size |

---

## Example Benchmark Report Output

```text
==================================================
   AURA VIDEO ENGINE BENCHMARK
==================================================
Input:
  Resolution:          1280x720
  Frames Received:     300
  Observed Input FPS:  59.88 FPS

Processing & Output:
  Frames Processed:    300
  Frames Displayed:    0
  Frames Dropped:      0
  Observed Output FPS: 59.88 FPS

Frame Timing (Processing Latency):
  Average Frame Time:  0.035 ms
  Median Frame Time:   0.031 ms
  P95 Frame Time:      0.048 ms
  Min / Max Time:      0.020 ms / 0.180 ms

Hardware Utilization:
  CPU Utilization:     12.4%
  RAM Utilization:     45.1%
  GPU Name:            NVIDIA GeForce RTX 4070
  GPU Utilization:     4.0%
  GPU Memory Used:     420.0 MB

Execution Summary:
  Total Runtime:       5.01 seconds
==================================================
```

---

## Running Unit Tests

Run the full pytest suite to verify capture handling, metrics calculations, and identity pipeline integrity:
```bash
pytest tests/ -v
```

---

## Known Limitations & Scope Boundaries

* **Phase 0 Scope**: Does not include neural super-resolution, AI enhancement models, YOLO object detection, MAVLink telemetry, or heavy GUI framework integration.
* **Hardware Stats**: GPU metrics require NVIDIA GPUs with `pynvml` installed; non-NVIDIA or unsupported environments report `"N/A"` without error.

---

## Next Steps (Phase 1 Preview)
Upon approval of Phase 0 baseline benchmark results, Phase 1 will introduce low-latency neural image enhancement modules plugged directly into the `BaseProcessor` interface.
