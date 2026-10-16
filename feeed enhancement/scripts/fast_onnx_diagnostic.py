"""Lightweight progress-logging benchmark for ONNX CPU and ONNX CUDA."""
import json
import os
import sys
import time

local_libs = os.path.abspath(".venv_libs")
if os.path.exists(local_libs) and local_libs not in sys.path:
    sys.path.insert(0, local_libs)

import numpy as np
import torch
import onnxruntime as ort
from app.enhancers.neural import ZeroDceEnhancer

def benchmark_single_provider(provider_name, warmup_frames=2, timed_frames=5):
    print(f"\n==================================================", flush=True)
    print(f"Testing Provider: {provider_name}", flush=True)
    print(f"==================================================", flush=True)

    available = ort.get_available_providers()
    print(f"Available Providers: {available}", flush=True)
    if provider_name not in available:
        print(f"STATUS: {provider_name} is NOT AVAILABLE", flush=True)
        return None

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    t_sess0 = time.perf_counter()
    try:
        session = ort.InferenceSession("outputs/models/zerodce.onnx", sess_options=so, providers=[provider_name])
    except Exception as e:
        print(f"Session Creation Failed: {e}", flush=True)
        return None
    t_sess1 = time.perf_counter()
    print(f"Session Created in {(t_sess1 - t_sess0)*1000:.2f} ms", flush=True)

    active = session.get_providers()
    print(f"Active Provider: {active}", flush=True)

    enhancer = ZeroDceEnhancer(model_path="outputs/models/Epoch99.pth", device="cpu", fp16=False, num_curves=8)
    np.random.seed(42)
    sample_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    print(f"Starting Warmup ({warmup_frames} frames)...", flush=True)
    for i in range(warmup_frames):
        t0 = time.perf_counter()
        tensor_in = enhancer.preprocess(sample_frame)
        _ = session.run(None, {"input": tensor_in.cpu().numpy()})[0]
        print(f"  Warmup frame {i+1}: {(time.perf_counter() - t0)*1000:.2f} ms", flush=True)

    print(f"Starting Timed Benchmarks ({timed_frames} frames)...", flush=True)
    latencies = []
    for i in range(timed_frames):
        t0 = time.perf_counter()
        tensor_in = enhancer.preprocess(sample_frame)
        t1 = time.perf_counter()
        ort_out = session.run(None, {"input": tensor_in.cpu().numpy()})[0]
        t2 = time.perf_counter()
        _ = enhancer.postprocess(torch.from_numpy(ort_out))
        t3 = time.perf_counter()

        prep = (t1 - t0) * 1000
        infer = (t2 - t1) * 1000
        post = (t3 - t2) * 1000
        tot = (t3 - t0) * 1000
        latencies.append(tot)
        print(f"  Timed frame {i+1}: Prep={prep:.2f}ms | Infer={infer:.2f}ms | Post={post:.2f}ms | Total={tot:.2f}ms", flush=True)

    mean_tot = float(np.mean(latencies))
    fps = 1000.0 / mean_tot
    print(f"COMPLETED: {provider_name} | Mean Total: {mean_tot:.2f} ms | FPS: {fps:.2f}", flush=True)
    return {"provider": provider_name, "mean_ms": mean_tot, "fps": fps}

if __name__ == "__main__":
    print("Checking CUDA provider first...", flush=True)
    cuda_res = benchmark_single_provider("CUDAExecutionProvider", warmup_frames=3, timed_frames=10)
    print("\nChecking CPU provider next...", flush=True)
    cpu_res = benchmark_single_provider("CPUExecutionProvider", warmup_frames=1, timed_frames=2)
