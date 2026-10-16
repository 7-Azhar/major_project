"""Small 5-warmup, 10-timed frame ONNX CUDA benchmark."""
import json
import os
import sys
import time

cudnn_dir = r"C:\Users\azhar\CODES\ai\.venv\Lib\site-packages\torch\lib"
cuda_dir = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
if os.path.exists(cudnn_dir):
    os.add_dll_directory(cudnn_dir)
if os.path.exists(cuda_dir):
    os.add_dll_directory(cuda_dir)

local_libs = os.path.abspath(".venv_libs")
if os.path.exists(local_libs) and local_libs not in sys.path:
    sys.path.insert(0, local_libs)

import numpy as np
import torch
import onnxruntime as ort
from app.enhancers.neural import ZeroDceEnhancer

def main():
    print("=" * 60)
    print("Phase 3A.2 Step 9 — 10-Frame ONNX CUDA Benchmark")
    print("=" * 60)

    onnx_path = "outputs/models/zerodce.onnx"
    enhancer = ZeroDceEnhancer(model_path="outputs/models/Epoch99.pth", device="cpu", fp16=False, num_curves=8)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_cuda = ort.InferenceSession(onnx_path, sess_options=so, providers=["CUDAExecutionProvider"])

    print(f"Active Provider: {session_cuda.get_providers()}")

    np.random.seed(42)
    sample_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    warmup_frames = 5
    timed_frames = 10

    print(f"\nRunning {warmup_frames} warm-up frames...")
    for _ in range(warmup_frames):
        tensor_in = enhancer.preprocess(sample_frame)
        _ = session_cuda.run(None, {"input": tensor_in.cpu().numpy()})[0]

    print(f"\nRunning {timed_frames} timed benchmark frames...")
    prep_times, infer_times, post_times, tot_times = [], [], [], []

    for i in range(timed_frames):
        t0 = time.perf_counter()
        tensor_in = enhancer.preprocess(sample_frame)
        t1 = time.perf_counter()
        ort_out = session_cuda.run(None, {"input": tensor_in.cpu().numpy()})[0]
        t2 = time.perf_counter()
        _ = enhancer.postprocess(torch.from_numpy(ort_out))
        t3 = time.perf_counter()

        prep = (t1 - t0) * 1000
        infer = (t2 - t1) * 1000
        post = (t3 - t2) * 1000
        tot = (t3 - t0) * 1000

        prep_times.append(prep)
        infer_times.append(infer)
        post_times.append(post)
        tot_times.append(tot)

        print(f"  Frame {i+1:2d}: Prep={prep:5.2f}ms | GPU Infer={infer:6.2f}ms | Post={post:5.2f}ms | Total={tot:6.2f}ms")

    tot_arr = np.array(tot_times)
    infer_arr = np.array(infer_times)

    print("\nBenchmark Summary (ONNX CUDA FP32):")
    print(f"  Median Total Latency:  {np.median(tot_arr):.2f} ms")
    print(f"  P95 Total Latency:     {np.percentile(tot_arr, 95):.2f} ms")
    print(f"  GPU Inference Median:  {np.median(infer_arr):.2f} ms")
    print(f"  Throughput FPS:        {1000.0 / np.mean(tot_arr):.2f} FPS")

    print("\n" + "=" * 60)
    print("10-Frame ONNX CUDA Benchmark Completed")
    print("=" * 60)

if __name__ == "__main__":
    main()
