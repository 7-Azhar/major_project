"""Reproducible PyTorch CPU Baseline Benchmark for Zero-DCE (Epoch99.pth)."""
import json
import os
import sys
import time
import numpy as np

# Ensure local libs directory is in path
local_libs = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv_libs"))
if os.path.exists(local_libs) and local_libs not in sys.path:
    sys.path.insert(0, local_libs)

import torch
from app.enhancers.neural import ZeroDceEnhancer

def run_pytorch_baseline():
    print("=" * 60)
    print("Phase 3A Step 3: Reproducible PyTorch CPU Baseline Benchmark")
    print("=" * 60)

    ckpt_path = os.path.join("outputs", "models", "Epoch99.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    enhancer = ZeroDceEnhancer(model_path=ckpt_path, device="cpu", fp16=False, num_curves=8)
    print(f"Loaded ZeroDceEnhancer from checkpoint: {ckpt_path}")

    # Generate fixed deterministic input: 1280x720 uint8 frame
    np.random.seed(42)
    sample_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    warmup_frames = 10
    timed_frames = 100

    print(f"Running {warmup_frames} warm-up frames...")
    for _ in range(warmup_frames):
        _ = enhancer.process(sample_frame)

    print(f"Running {timed_frames} timed benchmark frames...")
    preprocess_times = []
    forward_times = []
    postprocess_times = []
    total_times = []

    for _ in range(timed_frames):
        t0 = time.perf_counter()
        tensor_in = enhancer.preprocess(sample_frame)
        t1 = time.perf_counter()

        with torch.no_grad():
            tensor_out = enhancer._forward_neural(tensor_in)
        t2 = time.perf_counter()

        out_frame = enhancer.postprocess(tensor_out)
        t3 = time.perf_counter()

        prep = (t1 - t0) * 1000.0
        fwd = (t2 - t1) * 1000.0
        post = (t3 - t2) * 1000.0
        tot = (t3 - t0) * 1000.0

        preprocess_times.append(prep)
        forward_times.append(fwd)
        postprocess_times.append(post)
        total_times.append(tot)

    total_arr = np.array(total_times)
    fwd_arr = np.array(forward_times)
    prep_arr = np.array(preprocess_times)
    post_arr = np.array(postprocess_times)

    metrics = {
        "runtime": "PyTorch CPU",
        "precision": "FP32",
        "checkpoint": ckpt_path,
        "resolution": "1280x720",
        "warmup_frames": warmup_frames,
        "timed_frames": timed_frames,
        "preprocess_ms": {
            "mean": float(np.mean(prep_arr)),
            "median": float(np.median(prep_arr)),
            "p5": float(np.percentile(prep_arr, 5)),
            "p95": float(np.percentile(prep_arr, 95)),
            "min": float(np.min(prep_arr)),
            "max": float(np.max(prep_arr)),
        },
        "forward_ms": {
            "mean": float(np.mean(fwd_arr)),
            "median": float(np.median(fwd_arr)),
            "p5": float(np.percentile(fwd_arr, 5)),
            "p95": float(np.percentile(fwd_arr, 95)),
            "min": float(np.min(fwd_arr)),
            "max": float(np.max(fwd_arr)),
        },
        "postprocess_ms": {
            "mean": float(np.mean(post_arr)),
            "median": float(np.median(post_arr)),
            "p5": float(np.percentile(post_arr, 5)),
            "p95": float(np.percentile(post_arr, 95)),
            "min": float(np.min(post_arr)),
            "max": float(np.max(post_arr)),
        },
        "total_ms": {
            "mean": float(np.mean(total_arr)),
            "median": float(np.median(total_arr)),
            "p5": float(np.percentile(total_arr, 5)),
            "p95": float(np.percentile(total_arr, 95)),
            "min": float(np.min(total_arr)),
            "max": float(np.max(total_arr)),
        },
        "throughput_fps": float(1000.0 / np.mean(total_arr)),
    }

    print("\nBenchmark Results (PyTorch CPU FP32):")
    print(f"  Preprocess Mean:  {metrics['preprocess_ms']['mean']:.2f} ms")
    print(f"  Forward Mean:     {metrics['forward_ms']['mean']:.2f} ms")
    print(f"  Postprocess Mean: {metrics['postprocess_ms']['mean']:.2f} ms")
    print(f"  Total Mean:       {metrics['total_ms']['mean']:.2f} ms")
    print(f"  Total Median:     {metrics['total_ms']['median']:.2f} ms")
    print(f"  Total P95:        {metrics['total_ms']['p95']:.2f} ms")
    print(f"  Throughput FPS:   {metrics['throughput_fps']:.2f} FPS")

    os.makedirs("outputs", exist_ok=True)
    out_json = os.path.join("outputs", "phase3_pytorch_baseline.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved report to: {out_json}")
    print("=" * 60)

if __name__ == "__main__":
    run_pytorch_baseline()
