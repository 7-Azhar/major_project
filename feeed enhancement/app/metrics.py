"""Performance metrics calculation and system monitoring module for AURA Engine."""

import logging
import time
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Try importing optional system metrics dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False


class MetricsTracker:
    """Collects frame processing timings, frame counters, FPS, and system resource stats."""

    def __init__(self, resolution_w: int = 0, resolution_h: int = 0):
        self.resolution_w = resolution_w
        self.resolution_h = resolution_h

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        self.frames_received: int = 0
        self.frames_processed: int = 0
        self.frames_displayed: int = 0
        self.dropped_frames: int = 0

        self.frame_processing_times: List[float] = []  # in seconds
        self.inter_frame_intervals: List[float] = []   # in seconds
        self._last_frame_timestamp: Optional[float] = None

    def start(self):
        """Mark start of execution run."""
        self.start_time = time.perf_counter()
        self.end_time = None
        self._last_frame_timestamp = None

    def stop(self):
        """Mark end of execution run."""
        if self.end_time is None:
            self.end_time = time.perf_counter()

    def record_frame(
        self,
        processing_time_sec: float,
        timestamp_sec: Optional[float] = None,
        displayed: bool = True,
    ):
        """Record timing data for a processed frame."""
        if self.start_time is None:
            self.start()

        now = timestamp_sec if timestamp_sec is not None else time.perf_counter()

        if self._last_frame_timestamp is not None:
            interval = now - self._last_frame_timestamp
            if interval > 0:
                self.inter_frame_intervals.append(interval)

        self._last_frame_timestamp = now
        self.frame_processing_times.append(processing_time_sec)
        self.frames_processed += 1
        if displayed:
            self.frames_displayed += 1

    def update_capture_counts(self, frames_received: int, dropped_frames: int):
        """Update counts from capture thread."""
        self.frames_received = frames_received
        self.dropped_frames = dropped_frames

    @property
    def elapsed_runtime(self) -> float:
        """Returns total active runtime in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return max(0.0, end - self.start_time)

    def get_observed_input_fps(self) -> float:
        """Returns observed input FPS."""
        runtime = self.elapsed_runtime
        if runtime > 0 and self.frames_received > 0:
            return float(self.frames_received / runtime)
        return 0.0

    def get_current_stats(self) -> Dict[str, Any]:
        """Compute snapshot of performance metrics."""
        runtime = self.elapsed_runtime

        if runtime > 0:
            observed_input_fps = self.frames_received / runtime if self.frames_received > 0 else 0.0
            observed_output_fps = self.frames_processed / runtime if self.frames_processed > 0 else 0.0
        else:
            observed_input_fps = 0.0
            observed_output_fps = 0.0

        if self.frame_processing_times:
            times_ms = np.array(self.frame_processing_times) * 1000.0
            avg_frame_time_ms = float(np.mean(times_ms))
            median_frame_time_ms = float(np.median(times_ms))
            p95_frame_time_ms = float(np.percentile(times_ms, 95))
            min_frame_time_ms = float(np.min(times_ms))
            max_frame_time_ms = float(np.max(times_ms))
        else:
            avg_frame_time_ms = 0.0
            median_frame_time_ms = 0.0
            p95_frame_time_ms = 0.0
            min_frame_time_ms = 0.0
            max_frame_time_ms = 0.0

        hw_metrics = self.get_hardware_metrics()

        return {
            "resolution": f"{self.resolution_w}x{self.resolution_h}",
            "resolution_w": self.resolution_w,
            "resolution_h": self.resolution_h,
            "runtime_seconds": round(runtime, 3),
            "frames_received": self.frames_received,
            "frames_processed": self.frames_processed,
            "frames_displayed": self.frames_displayed,
            "frames_dropped": self.dropped_frames,
            "observed_input_fps": round(observed_input_fps, 2),
            "observed_output_fps": round(observed_output_fps, 2),
            "avg_frame_time_ms": round(avg_frame_time_ms, 3),
            "median_frame_time_ms": round(median_frame_time_ms, 3),
            "p95_frame_time_ms": round(p95_frame_time_ms, 3),
            "min_frame_time_ms": round(min_frame_time_ms, 3),
            "max_frame_time_ms": round(max_frame_time_ms, 3),
            "hardware": hw_metrics,
        }

    def get_hardware_metrics(self) -> Dict[str, Any]:
        """Fetch system CPU, RAM, and GPU statistics safely without failing."""
        metrics = {
            "cpu_utilization_pct": "N/A",
            "ram_utilization_pct": "N/A",
            "process_rss_mb": "N/A",
            "gpu_utilization_pct": "N/A",
            "gpu_memory_used_mb": "N/A",
            "gpu_name": "N/A",
        }

        if HAS_PSUTIL:
            try:
                metrics["cpu_utilization_pct"] = psutil.cpu_percent(interval=None)
                metrics["ram_utilization_pct"] = psutil.virtual_memory().percent
                metrics["process_rss_mb"] = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
            except Exception as e:
                logger.debug(f"Failed collecting CPU/RAM metrics: {e}")

        if HAS_NVML:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(gpu_name, bytes):
                        gpu_name = gpu_name.decode("utf-8")
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    metrics["gpu_name"] = gpu_name
                    metrics["gpu_utilization_pct"] = utilization.gpu
                    metrics["gpu_memory_used_mb"] = round(mem_info.used / (1024 * 1024), 1)
            except Exception as e:
                logger.debug(f"Failed collecting NVML GPU metrics: {e}")

        return metrics

    def generate_report_text(self, title: str = "AURA VIDEO ENGINE BENCHMARK") -> str:
        """Format metrics snapshot into human-readable benchmark report."""
        stats = self.get_current_stats()
        hw = stats["hardware"]

        report = [
            "=" * 50,
            f"   {title}",
            "=" * 50,
            "Input:",
            f"  Resolution:          {stats['resolution']}",
            f"  Frames Received:     {stats['frames_received']}",
            f"  Observed Input FPS:  {stats['observed_input_fps']} FPS",
            "",
            "Processing & Output:",
            f"  Frames Processed:    {stats['frames_processed']}",
            f"  Frames Displayed:    {stats['frames_displayed']}",
            f"  Frames Dropped:      {stats['frames_dropped']}",
            f"  Observed Output FPS: {stats['observed_output_fps']} FPS",
            "",
            "Frame Timing (Processing Latency):",
            f"  Average Frame Time:  {stats['avg_frame_time_ms']} ms",
            f"  Median Frame Time:   {stats['median_frame_time_ms']} ms",
            f"  P95 Frame Time:      {stats['p95_frame_time_ms']} ms",
            f"  Min / Max Time:      {stats['min_frame_time_ms']} ms / {stats['max_frame_time_ms']} ms",
            "",
            "Hardware Utilization:",
            f"  CPU Utilization:     {hw['cpu_utilization_pct']}%" if hw['cpu_utilization_pct'] != "N/A" else "  CPU Utilization:     N/A",
            f"  RAM Utilization:     {hw['ram_utilization_pct']}%" if hw['ram_utilization_pct'] != "N/A" else "  RAM Utilization:     N/A",
            f"  GPU Name:            {hw['gpu_name']}",
            f"  GPU Utilization:     {hw['gpu_utilization_pct']}%" if hw['gpu_utilization_pct'] != "N/A" else "  GPU Utilization:     N/A",
            f"  GPU Memory Used:     {hw['gpu_memory_used_mb']} MB" if hw['gpu_memory_used_mb'] != "N/A" else "  GPU Memory Used:     N/A",
            "",
            "Execution Summary:",
            f"  Total Runtime:       {stats['runtime_seconds']} seconds",
            "=" * 50,
        ]
        return "\n".join(report)
