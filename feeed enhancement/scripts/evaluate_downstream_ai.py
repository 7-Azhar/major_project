"""Downstream AI Machine Perception Evaluation Suite for AURA Phase 2.

Evaluates how raw FPV video vs classical & neural enhancement methods impact downstream
computer vision (YOLO object detection) performance.

Measures:
- Latency (ms) & Output FPS
- Full-Reference & No-Reference Quality Metrics (PSNR, SSIM, Tenengrad Gain %)
- Machine Perception Metrics (Precision, Recall, mAP@0.5, Avg Confidence, False Positives)

Saves machine-readable research evaluation JSON artifacts for research comparison.
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
from app.eval import YoloEvaluator
from app.quality import evaluate_quality_pair

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_phase2_evaluation_suite(
    source: str = "samples/test_synthetic.mp4",
    duration_sec: float = 3.0,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """Runs comprehensive downstream AI evaluation across all enhancement baselines."""
    os.makedirs(output_dir, exist_ok=True)

    processors_to_eval = [
        ("Baseline 0: Identity", "identity"),
        ("Baseline 1: CLAHE", "clahe"),
        ("Baseline 2: Unsharp Mask", "unsharp"),
        ("Baseline 3: Bilateral Denoise", "bilateral"),
        ("Baseline 4: Composite", "composite"),
        ("Baseline 5: Neural Zero-DCE", "zerodce"),
    ]

    # Collect sample stream frames for downstream AI evaluation (sampled for fast CPU evaluation)
    cap = cv2.VideoCapture(source)
    raw_frames: List[np.ndarray] = []
    max_eval_frames = min(20, max(5, int(10 * duration_sec)))

    while cap.isOpened() and len(raw_frames) < max_eval_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        raw_frames.append(frame)
    cap.release()

    if not raw_frames:
        raise ValueError(f"Could not read evaluation frames from source: {source}")

    yolo_evaluator = YoloEvaluator()
    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 115)
    print("      AURA ENGINE PHASE 2 — DOWNSTREAM AI MACHINE PERCEPTION EVALUATION SUITE")
    print("=" * 115)

    for label, proc_name in processors_to_eval:
        logger.info(f"\n--- Evaluating {label} ({proc_name}) ---")

        cfg = EngineConfig(
            source=source,
            max_frames=max_eval_frames,
            no_display=True,
            loop=True,
            processor=proc_name,
            output_dir=output_dir,
        )

        runner = BenchmarkRunner(config=cfg)
        run_data = runner.run()

        proc_instance = get_enhancer_by_name(proc_name)

        # Process offline stream copy for quality & downstream AI evaluation
        enhanced_frames: List[np.ndarray] = []
        for rf in raw_frames:
            ef, _ = proc_instance.process(rf)
            enhanced_frames.append(ef)

        # 1. Quality metrics
        q_metrics = evaluate_quality_pair(raw_frames[0], enhanced_frames[0])

        # 2. Downstream AI perception metrics
        detection_eval = yolo_evaluator.evaluate_stream(
            raw_frames=raw_frames,
            enhanced_frames=enhanced_frames,
            processor_name=proc_instance.__class__.__name__,
        )

        run_summary = {
            "baseline_label": label,
            "processor": proc_name,
            "processor_class": proc_instance.__class__.__name__,
            "observed_output_fps": run_data.get("observed_output_fps"),
            "avg_frame_time_ms": run_data.get("avg_frame_time_ms"),
            "p95_frame_time_ms": run_data.get("p95_frame_time_ms"),
            "quality_metrics": q_metrics,
            "detection_metrics": detection_eval.to_dict(),
            "hardware": run_data.get("hardware"),
        }

        results.append(run_summary)

    # Print Machine Perception & Performance Comparison Matrix
    print("\n" + "=" * 135)
    print("                                   PHASE 2 EXPERIMENTAL EVALUATION MATRIX")
    print("=" * 135)
    header = (
        f"| {'Baseline / Processor':<25} | {'Output FPS':<10} | {'Avg Latency':<12} | "
        f"{'PSNR (dB)':<10} | {'SSIM':<7} | {'mAP@0.5':<8} | {'Precision':<9} | {'Recall':<8} | {'Conf Delta':<11} |"
    )
    divider = (
        "|-" + "-|-".join(["-"*25, "-"*10, "-"*12, "-"*10, "-"*7, "-"*8, "-"*9, "-"*8, "-"*11]) + "-|"
    )
    print(header)
    print(divider)

    for r in results:
        proc_str = r["baseline_label"]
        fps_str = f"{r['observed_output_fps']:.1f}"
        avg_lat = f"{r['avg_frame_time_ms']:.3f} ms"
        qm = r["quality_metrics"]
        dm = r["detection_metrics"]

        psnr_str = f"{qm['psnr_db']:.2f}" if qm['psnr_db'] != 999.0 else "INF"
        ssim_str = f"{qm['ssim']:.4f}"
        map_str = f"{dm['map_50']:.4f}"
        prec_str = f"{dm['precision']:.4f}"
        rec_str = f"{dm['recall']:.4f}"
        conf_str = f"{dm['confidence_delta_pct']:+.1f}%"

        print(
            f"| {proc_str:<25} | {fps_str:<10} | {avg_lat:<12} | "
            f"{psnr_str:<10} | {ssim_str:<7} | {map_str:<8} | {prec_str:<9} | {rec_str:<8} | {conf_str:<11} |"
        )

    print("=" * 135)

    # Export machine readable evaluation report JSON
    timestamp = int(time.time())
    json_path = os.path.join(output_dir, f"phase2_eval_report_{timestamp}.json")
    latest_json_path = os.path.join(output_dir, "phase2_eval_report.json")

    suite_export = {
        "suite": "AURA Phase 2 Downstream AI & Quality Evaluation",
        "timestamp": timestamp,
        "source": source,
        "benchmark_duration_per_processor_sec": duration_sec,
        "results": results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(suite_export, f, indent=2)
    with open(latest_json_path, "w", encoding="utf-8") as f:
        json.dump(suite_export, f, indent=2)

    logger.info(f"\nSaved machine-readable evaluation report JSON: {json_path}")
    logger.info(f"Saved latest evaluation report JSON: {latest_json_path}")

    return suite_export


if __name__ == "__main__":
    import numpy as np
    args = parse_args()
    duration = args.benchmark_duration if args.benchmark_duration > 0 else 3.0
    run_phase2_evaluation_suite(source=args.source, duration_sec=duration, output_dir=args.output_dir)
