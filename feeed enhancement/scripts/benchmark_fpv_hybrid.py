"""Phase 5A — Hybrid FPV Restoration Prototype Benchmarking Script.

Evaluates:
1. Raw (Identity)
2. Bilateral
3. FPV Restoration Baseline
4. FPV Hybrid (NEW)
5. Zero-DCE FP16

Measures:
- Latency (median, mean, p95)
- Effective FPS
- CPU & RAM usage
- Temporal stability (interframe flicker ratio)
- Generates outputs/fpv_analysis/hybrid_benchmark.json, csv, and contact sheet comparison.
"""

import csv
import json
import os
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np
import psutil

# CUDA DLL Setup
cudnn_dir = r"C:\Users\azhar\CODES\ai\.venv\Lib\site-packages\torch\lib"
cuda_dir = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
if os.path.exists(cudnn_dir):
    os.add_dll_directory(cudnn_dir)
if os.path.exists(cuda_dir):
    os.add_dll_directory(cuda_dir)

local_libs = os.path.abspath(".venv_libs")
if os.path.exists(local_libs) and local_libs not in sys.path:
    sys.path.insert(0, local_libs)

from app.enhancers import (
    BilateralDenoiseEnhancer,
    FpvHybridEnhancer,
    FpvRestorationEnhancer,
    IdentityProcessor,
)
from app.manual_viewer import ZeroDceOnnxCUDAEnhancer


def compute_frame_motion(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> float:
    """Compute frame-to-frame motion magnitude using mean absolute pixel difference."""
    g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(np.abs(g1.astype(np.float32) - g2.astype(np.float32))))


def create_contact_sheet(image_items: List[tuple], output_path: str, grid_cols: int = 3, target_h: int = 240):
    """Create a contact sheet grid from images with labels."""
    loaded = []
    for label, img in image_items:
        if img is not None:
            scale = target_h / img.shape[0]
            w = int(img.shape[1] * scale)
            img_r = cv2.resize(img, (w, target_h))

            # Header label
            bar = np.zeros((30, w, 3), dtype=np.uint8)
            cv2.putText(bar, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
            labeled = np.vstack([bar, img_r])
            loaded.append(labeled)

    if not loaded:
        return

    max_w = max(im.shape[1] for im in loaded)
    padded = []
    for im in loaded:
        if im.shape[1] < max_w:
            pad = np.zeros((im.shape[0], max_w - im.shape[1], 3), dtype=np.uint8)
            im = np.hstack([im, pad])
        padded.append(im)

    rows = []
    for i in range(0, len(padded), grid_cols):
        chunk = padded[i : i + grid_cols]
        while len(chunk) < grid_cols:
            blank = np.zeros_like(padded[0])
            chunk.append(blank)
        rows.append(np.hstack(chunk))

    grid = np.vstack(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, grid)
    print(f"Saved comparison contact sheet: {output_path}", flush=True)


def main():
    print("=" * 70, flush=True)
    print("AURA Phase 5A — Hybrid FPV Restoration Prototype Benchmark", flush=True)
    print("=" * 70, flush=True)

    video_path = os.path.join("samples", "fpv", "real_fpv_user_flight.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join("samples", "real_fpv_degraded.mp4")

    print(f"Target FPV Feed: {video_path}", flush=True)

    # Load 150 frames for benchmark
    cap = cv2.VideoCapture(video_path)
    clip_frames = []
    for _ in range(150):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        clip_frames.append(frame)
    cap.release()

    print(f"Loaded {len(clip_frames)} frames for evaluation.", flush=True)

    # Compute raw baseline motion
    raw_diffs = []
    for i in range(len(clip_frames) - 1):
        raw_diffs.append(compute_frame_motion(clip_frames[i], clip_frames[i + 1]))
    raw_motion_mean = float(np.mean(raw_diffs))

    # Processors to compare
    processors = {
        "Raw (Identity)": IdentityProcessor(),
        "Bilateral": BilateralDenoiseEnhancer(),
        "FPV Restoration": FpvRestorationEnhancer(),
        "FPV Hybrid (NEW)": FpvHybridEnhancer(
            bilateral_strength=1.0,
            restoration_strength=1.0,
            contrast_strength=0.2,
            detail_strength=0.2,
        ),
        "Zero-DCE FP16": ZeroDceOnnxCUDAEnhancer(),
    }

    process_info = psutil.Process()
    results = []
    comparison_frames = []

    print("\nRunning benchmark across processors...", flush=True)
    print("-" * 105, flush=True)
    print(f"{'Processor':<25} | {'Med ms':>7} | {'Mean ms':>7} | {'P95 ms':>7} | {'FPS':>6} | {'Flicker':>7} | {'Status (<33.3ms)':<16}", flush=True)
    print("-" * 105, flush=True)

    for name, processor in processors.items():
        latencies = []
        enhanced_frames = []

        # Memory/CPU snapshot before run
        mem_before = process_info.memory_info().rss / (1024 * 1024)

        for f in clip_frames:
            t0 = time.perf_counter()
            if hasattr(processor, "process"):
                ef, _ = processor.process(f)
            else:
                ef = f.copy()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
            enhanced_frames.append(ef)

        mem_after = process_info.memory_info().rss / (1024 * 1024)
        cpu_pct = process_info.cpu_percent()

        # Compute interframe differences
        enh_diffs = []
        for i in range(len(enhanced_frames) - 1):
            enh_diffs.append(compute_frame_motion(enhanced_frames[i], enhanced_frames[i + 1]))

        flicker_ratio = float(np.mean(enh_diffs) / (raw_motion_mean + 1e-5))
        med_lat = float(np.median(latencies))
        mean_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        eff_fps = float(1000.0 / med_lat) if med_lat > 0 else 999.0

        status = "PASSED (<33.3ms)" if med_lat <= 33.3 else f"FAILED (+{med_lat - 33.3:.1f}ms)"

        item = {
            "Processor": name,
            "Median_Latency_ms": round(med_lat, 2),
            "Mean_Latency_ms": round(mean_lat, 2),
            "P95_Latency_ms": round(p95_lat, 2),
            "Effective_FPS": round(eff_fps, 1),
            "Temporal_Flicker_Ratio": round(flicker_ratio, 3),
            "CPU_Percent": round(cpu_pct, 1),
            "RAM_MB": round(mem_after, 1),
            "Status_30FPS": status,
        }
        results.append(item)
        comparison_frames.append((name, enhanced_frames[15]))

        print(f"{name:<25} | {med_lat:7.2f} | {mean_lat:7.2f} | {p95_lat:7.2f} | {eff_fps:6.1f} | {flicker_ratio:7.3f} | {status:<16}", flush=True)

    print("-" * 105, flush=True)

    # Save JSON & CSV reports
    os.makedirs("outputs/fpv_analysis", exist_ok=True)
    out_json = "outputs/fpv_analysis/hybrid_benchmark.json"
    out_csv = "outputs/fpv_analysis/hybrid_benchmark.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved benchmark JSON: {out_json}", flush=True)
    print(f"Saved benchmark CSV:  {out_csv}", flush=True)

    # Save contact sheet comparison
    out_img = "outputs/fpv_analysis/hybrid_comparison.jpg"
    create_contact_sheet(comparison_frames, out_img, grid_cols=3, target_h=240)


if __name__ == "__main__":
    main()
