"""Comprehensive ONNX Runtime Benchmark Script (CPU, CUDA, FP16)."""
import json
import os
import sys
import time

local_libs = os.path.abspath(".venv_libs")
if os.path.exists(local_libs) and local_libs not in sys.path:
    sys.path.insert(0, local_libs)

import cv2
import numpy as np
import torch

try:
    import onnxruntime as ort
except ImportError as e:
    print(f"Error importing onnxruntime: {e}")
    sys.exit(1)

from app.enhancers.neural import ZeroDceEnhancer

def benchmark_provider(provider_name, model_path="outputs/models/zerodce.onnx", warmup_frames=10, timed_frames=50, precision="FP32"):
    print(f"\n--- Benchmarking Provider: {provider_name} ({precision}) ---")
    available_providers = ort.get_available_providers()
    print(f"  Available Providers in ORT: {available_providers}")
    print(f"  Requested Provider: {provider_name}")

    if provider_name not in available_providers:
        print(f"  ERROR: Provider {provider_name} is NOT available in ONNX Runtime build!")
        return None

    # Explicit session options and provider options
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        session = ort.InferenceSession(model_path, sess_options=so, providers=[provider_name])
    except Exception as e:
        print(f"  Failed to create InferenceSession for {provider_name}: {e}")
        return None

    active_providers = session.get_providers()
    print(f"  Active Session Provider(s): {active_providers}")

    # Initialize enhancer for preprocessing and postprocessing logic
    enhancer = ZeroDceEnhancer(model_path="outputs/models/Epoch99.pth", device="cpu", fp16=False, num_curves=8)

    np.random.seed(42)
    sample_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    # Warm-up pass
    print(f"  Running {warmup_frames} warm-up iterations...")
    for _ in range(warmup_frames):
        tensor_in = enhancer.preprocess(sample_frame)
        input_data = tensor_in.cpu().numpy()
        _ = session.run(None, {"input": input_data})[0]

    # Timed benchmark loop
    print(f"  Running {timed_frames} timed benchmark iterations...")
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    total_times = []

    for _ in range(timed_frames):
        t0 = time.perf_counter()
        tensor_in = enhancer.preprocess(sample_frame)
        t1 = time.perf_counter()

        input_data = tensor_in.cpu().numpy()
        ort_out = session.run(None, {"input": input_data})[0]
        t2 = time.perf_counter()

        out_tensor = torch.from_numpy(ort_out)
        _ = enhancer.postprocess(out_tensor)
        t3 = time.perf_counter()

        prep = (t1 - t0) * 1000.0
        infer = (t2 - t1) * 1000.0
        post = (t3 - t2) * 1000.0
        tot = (t3 - t0) * 1000.0

        preprocess_times.append(prep)
        inference_times.append(infer)
        postprocess_times.append(post)
        total_times.append(tot)

    prep_arr = np.array(preprocess_times)
    infer_arr = np.array(inference_times)
    post_arr = np.array(postprocess_times)
    tot_arr = np.array(total_times)

    metrics = {
        "runtime": f"ONNX {provider_name.replace('ExecutionProvider', '')}",
        "provider": provider_name,
        "active_providers": active_providers,
        "precision": precision,
        "model_path": model_path,
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
        "inference_ms": {
            "mean": float(np.mean(infer_arr)),
            "median": float(np.median(infer_arr)),
            "p5": float(np.percentile(infer_arr, 5)),
            "p95": float(np.percentile(infer_arr, 95)),
            "min": float(np.min(infer_arr)),
            "max": float(np.max(infer_arr)),
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
            "mean": float(np.mean(tot_arr)),
            "median": float(np.median(tot_arr)),
            "p5": float(np.percentile(tot_arr, 5)),
            "p95": float(np.percentile(tot_arr, 95)),
            "min": float(np.min(tot_arr)),
            "max": float(np.max(tot_arr)),
        },
        "throughput_fps": float(1000.0 / np.mean(tot_arr)),
    }

    print(f"  Preprocess Mean:  {metrics['preprocess_ms']['mean']:.2f} ms")
    print(f"  Inference Mean:   {metrics['inference_ms']['mean']:.2f} ms")
    print(f"  Postprocess Mean: {metrics['postprocess_ms']['mean']:.2f} ms")
    print(f"  Total Mean:       {metrics['total_ms']['mean']:.2f} ms")
    print(f"  Total Median:     {metrics['total_ms']['median']:.2f} ms")
    print(f"  Total P95:        {metrics['total_ms']['p95']:.2f} ms")
    print(f"  Throughput FPS:   {metrics['throughput_fps']:.2f} FPS")

    return metrics

def main():
    print("=" * 60)
    print("Phase 3A Step 4, 5, 7: ONNX Runtime Benchmarking Suite")
    print("=" * 60)

    # 1. ONNX CPU Benchmark
    cpu_metrics = benchmark_provider("CPUExecutionProvider", warmup_frames=5, timed_frames=10)
    if cpu_metrics:
        with open("outputs/phase3_onnx_cpu.json", "w", encoding="utf-8") as f:
            json.dump(cpu_metrics, f, indent=2)

    # 2. ONNX CUDA Benchmark (FP32)
    cuda_metrics = benchmark_provider("CUDAExecutionProvider", warmup_frames=10, timed_frames=50)
    if cuda_metrics:
        with open("outputs/phase3_onnx_cuda.json", "w", encoding="utf-8") as f:
            json.dump(cuda_metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("ONNX Benchmarks Completed")
    print("=" * 60)

if __name__ == "__main__":
    main()
