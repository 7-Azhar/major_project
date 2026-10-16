"""Comparative Benchmark Suite for AURA Phase 1 Enhancers.

Runs automated comparative benchmarks across all Phase 1 baselines:
1. Baseline 0: Identity (Pass-through)
2. Baseline 1: CLAHE (Illumination / Contrast)
3. Baseline 2: Unsharp Mask (Sharpening)
4. Baseline 3: Bilateral Denoise (Denoising)
5. Baseline 4: Composite Pipeline (Denoise -> CLAHE -> Sharpen)

Measures and records FPS, processing latency (Avg, Median, P95), dropped frames,
objective quality metrics (PSNR, SSIM, Tenengrad sharpness gain %), and saves
structured machine-readable JSON artifacts for paper-grade research comparison.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from app.benchmark import BenchmarkRunner
from app.config import EngineConfig, parse_args
from app.enhancers import get_enhancer_by_name
from app.quality import evaluate_quality_pair

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_comparative_suite(
    source: str = "samples/test_synthetic.mp4",
    duration_sec: float = 3.0,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """Runs comparative benchmarks across all Phase 1 baseline enhancers."""
    os.makedirs(output_dir, exist_ok=True)

    processors_to_test = [
        ("Baseline 0: Identity", "identity"),
        ("Baseline 1: CLAHE", "clahe"),
        ("Baseline 2: Unsharp Mask", "unsharp"),
        ("Baseline 3: Bilateral Denoise", "bilateral"),
        ("Baseline 4: Composite", "composite"),
    ]

    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("      AURA ENGINE PHASE 1 — COMPARATIVE ENHANCEMENT BENCHMARK SUITE")
    print("=" * 80)

    # Read sample reference frames to calculate PSNR/SSIM/Tenengrad for each processor
    cap = cv2.VideoCapture(source)
    ret, sample_raw_frame = cap.read()
    cap.release()

    if not ret or sample_raw_frame is None:
        raise ValueError(f"Could not read reference sample frame from source: {source}")

    for label, proc_name in processors_to_test:
        logger.info(f"\n--- Testing {label} ({proc_name}) ---")

        cfg = EngineConfig(
            source=source,
            benchmark_duration=duration_sec,
            no_display=True,
            loop=True,
            processor=proc_name,
            output_dir=output_dir,
        )

        runner = BenchmarkRunner(config=cfg)
        run_data = runner.run()

        # Compute sample quality metrics against reference raw frame
        proc_instance = get_enhancer_by_name(proc_name)
        enhanced_sample, _ = proc_instance.process(sample_raw_frame)
        q_metrics = evaluate_quality_pair(sample_raw_frame, enhanced_sample)

        run_summary = {
            "baseline_label": label,
            "processor": proc_name,
            "processor_class": proc_instance.__class__.__name__,
            "resolution": run_data.get("resolution"),
            "frames_processed": run_data.get("frames_processed"),
            "frames_dropped": run_data.get("frames_dropped"),
            "observed_input_fps": run_data.get("observed_input_fps"),
            "observed_output_fps": run_data.get("observed_output_fps"),
            "avg_frame_time_ms": run_data.get("avg_frame_time_ms"),
            "median_frame_time_ms": run_data.get("median_frame_time_ms"),
            "p95_frame_time_ms": run_data.get("p95_frame_time_ms"),
            "min_frame_time_ms": run_data.get("min_frame_time_ms"),
            "max_frame_time_ms": run_data.get("max_frame_time_ms"),
            "quality_metrics": q_metrics,
            "hardware": run_data.get("hardware"),
        }

        results.append(run_summary)

    # Print Comparative Performance Table
    print("\n" + "=" * 105)
    print("                                PHASE 1 PERFORMANCE COMPARISON MATRIX")
    print("=" * 105)
    header = f"| {'Baseline / Processor':<25} | {'Output FPS':<10} | {'Avg Latency':<12} | {'P95 Latency':<12} | {'PSNR (dB)':<10} | {'SSIM':<7} | {'Tenengrad Gain':<14} |"
    divider = "|-" + "-|-".join(["-"*25, "-"*10, "-"*12, "-"*12, "-"*10, "-"*7, "-"*14]) + "-|"
    print(header)
    print(divider)

    for r in results:
        proc_str = r["baseline_label"]
        fps_str = f"{r['observed_output_fps']:.1f}"
        avg_lat = f"{r['avg_frame_time_ms']:.3f} ms"
        p95_lat = f"{r['p95_frame_time_ms']:.3f} ms"
        qm = r["quality_metrics"]
        psnr_str = f"{qm['psnr_db']:.2f}" if qm['psnr_db'] != 999.0 else "INF (Identical)"
        ssim_str = f"{qm['ssim']:.4f}"
        tenengrad_str = f"{qm['sharpness_gain_pct']:+.1f}%"

        print(f"| {proc_str:<25} | {fps_str:<10} | {avg_lat:<12} | {p95_lat:<12} | {psnr_str:<10} | {ssim_str:<7} | {tenengrad_str:<14} |")

    print("=" * 105)

    # Export machine readable combined comparison JSON
    timestamp = int(time.time())
    json_path = os.path.join(output_dir, f"phase1_comparative_benchmark_{timestamp}.json")
    latest_json_path = os.path.join(output_dir, "phase1_comparative_benchmark.json")

    suite_export = {
        "suite": "AURA Phase 1 Comparative Benchmark",
        "timestamp": timestamp,
        "source": source,
        "benchmark_duration_per_processor_sec": duration_sec,
        "results": results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(suite_export, f, indent=2)
    with open(latest_json_path, "w", encoding="utf-8") as f:
        json.dump(suite_export, f, indent=2)

    logger.info(f"\nSaved machine-readable comparative benchmark JSON: {json_path}")
    logger.info(f"Saved latest comparative benchmark JSON: {latest_json_path}")

    return suite_export


if __name__ == "__main__":
    args = parse_args()
    duration = args.benchmark_duration if args.benchmark_duration > 0 else 3.0
    run_comparative_suite(source=args.source, duration_sec=duration, output_dir=args.output_dir)
