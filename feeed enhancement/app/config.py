"""Configuration settings and argument parsing for AURA Video Enhancement Engine."""

import argparse
from dataclasses import dataclass
import os
from typing import Union


@dataclass
class EngineConfig:
    """Holds configuration parameters for video capture, processing, and benchmarking."""

    source: Union[int, str] = 0
    requested_width: int = 0
    requested_height: int = 0
    requested_fps: float = 0.0
    benchmark_duration: float = 0.0  # In seconds, 0 means unlimited
    max_frames: int = 0  # 0 means unlimited
    output_dir: str = "outputs"
    log_dir: str = "logs"
    record: bool = False
    no_display: bool = False
    queue_size: int = 2  # Bounded buffer size for low-latency FPV streaming
    loop: bool = False
    processor: str = "identity"
    side_by_side: bool = False
    compute_quality: bool = False
    clahe_clip: float = 2.0
    unsharp_amount: float = 1.5
    device: str = "auto"
    record_filename: str = ""

    def __post_init__(self):
        """Ensure directories exist."""
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)


def parse_args(args_list=None) -> EngineConfig:
    """Parse command-line arguments into EngineConfig."""
    parser = argparse.ArgumentParser(
        description="AURA Video Enhancement Engine - Phase 0 Benchmark & Streaming"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: camera index (e.g. 0) or file path (e.g. samples/test.mp4)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Requested capture width (0 for native)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Requested capture height (0 for native)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Requested capture FPS (0 for native)",
    )
    parser.add_argument(
        "--benchmark-duration",
        type=float,
        default=0.0,
        help="Duration in seconds for benchmark mode (0 for continuous)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process before stopping (0 for unlimited)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save benchmark reports and output videos",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save log files",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Enable recording of processed video output to disk",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable live video display window (useful for headless benchmarking)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=2,
        help="Maximum buffer queue size for capture thread (bounded low latency)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop input video file continuously when end of stream is reached",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default="identity",
        help="Enhancement processor: identity, clahe, unsharp, bilateral, composite",
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Display raw input and enhanced output side-by-side",
    )
    parser.add_argument(
        "--compute-quality",
        action="store_true",
        help="Compute PSNR, SSIM, and Tenengrad sharpness quality metrics",
    )
    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=2.0,
        help="CLAHE contrast clip limit (default: 2.0)",
    )
    parser.add_argument(
        "--unsharp-amount",
        type=float,
        default=1.5,
        help="Unsharp mask sharpening amount (default: 1.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device placement for neural models: auto, cpu, cuda",
    )

    parsed = parser.parse_args(args_list)

    # Convert numeric source string to int if applicable
    source_val: Union[int, str] = parsed.source
    if parsed.source.isdigit():
        source_val = int(parsed.source)

    return EngineConfig(
        source=source_val,
        requested_width=parsed.width,
        requested_height=parsed.height,
        requested_fps=parsed.fps,
        benchmark_duration=parsed.benchmark_duration,
        max_frames=parsed.max_frames,
        output_dir=parsed.output_dir,
        log_dir=parsed.log_dir,
        record=parsed.record,
        no_display=parsed.no_display,
        queue_size=parsed.queue_size,
        loop=parsed.loop,
        processor=parsed.processor,
        side_by_side=parsed.side_by_side,
        compute_quality=parsed.compute_quality,
        clahe_clip=parsed.clahe_clip,
        unsharp_amount=parsed.unsharp_amount,
        device=parsed.device,
    )
