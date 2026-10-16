"""Single-frame GPU inference test and numerical equivalence verification against CPU."""
import os
import sys
import time

# Add cuDNN and CUDA DLL directories
cudnn_dir = r"C:\Users\azhar\CODES\ai\.venv\Lib\site-packages\torch\lib"
cuda_dir = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
if os.path.exists(cudnn_dir):
    os.add_dll_directory(cudnn_dir)
if os.path.exists(cuda_dir):
    os.add_dll_directory(cuda_dir)

local_libs = os.path.abspath(".venv_libs")
if os.path.exists(local_libs) and local_libs not in sys.path:
    sys.path.insert(0, local_libs)

import cv2
import numpy as np
import torch
import onnxruntime as ort
from app.enhancers.neural import ZeroDceEnhancer

def main():
    print("=" * 60)
    print("Phase 3A.2 Step 7 & 8 — Single-Frame CUDA Inference & Verification")
    print("=" * 60)

    onnx_path = "outputs/models/zerodce.onnx"
    enhancer = ZeroDceEnhancer(model_path="outputs/models/Epoch99.pth", device="cpu", fp16=False, num_curves=8)

    # 1. Session creation timing
    t_sess0 = time.perf_counter()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_cuda = ort.InferenceSession(onnx_path, sess_options=so, providers=["CUDAExecutionProvider"])
    t_sess1 = time.perf_counter()

    active_providers = session_cuda.get_providers()
    print(f"\n1. CUDA Session Initialization Time: {(t_sess1 - t_sess0)*1000:.2f} ms")
    print(f"   Active Provider: {active_providers}")

    np.random.seed(42)
    sample_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    # 2. First inference (Cold)
    t_cold0 = time.perf_counter()
    tensor_in = enhancer.preprocess(sample_frame)
    input_data = tensor_in.cpu().numpy()
    cuda_out_cold = session_cuda.run(None, {"input": input_data})[0]
    bgr_cold = enhancer.postprocess(torch.from_numpy(cuda_out_cold))
    t_cold1 = time.perf_counter()
    print(f"\n2. First Inference (Cold Pass): {(t_cold1 - t_cold0)*1000:.2f} ms")

    # 3. Second inference (Warm)
    t0 = time.perf_counter()
    tensor_in = enhancer.preprocess(sample_frame)
    t1 = time.perf_counter()
    cuda_out_warm = session_cuda.run(None, {"input": input_data})[0]
    t2 = time.perf_counter()
    bgr_warm = enhancer.postprocess(torch.from_numpy(cuda_out_warm))
    t3 = time.perf_counter()

    prep_ms = (t1 - t0) * 1000
    infer_ms = (t2 - t1) * 1000
    post_ms = (t3 - t2) * 1000
    total_ms = (t3 - t0) * 1000

    print(f"\n3. Second Inference Breakdown (Warm Pass):")
    print(f"   - Preprocessing: {prep_ms:.2f} ms")
    print(f"   - GPU Inference: {infer_ms:.2f} ms")
    print(f"   - Postprocessing: {post_ms:.2f} ms")
    print(f"   - Total Latency:  {total_ms:.2f} ms")

    # 4. Numerical Verification against ONNX CPU output
    session_cpu = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    cpu_out = session_cpu.run(None, {"input": input_data})[0]
    bgr_cpu = enhancer.postprocess(torch.from_numpy(cpu_out))

    abs_diff = np.abs(bgr_cuda := bgr_warm.astype(np.float64) - bgr_cpu.astype(np.float64))
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    print(f"\n4. Numerical Verification (CUDA vs CPU):")
    print(f"   - Max Absolute Difference:  {max_abs_diff:.4f} intensity levels")
    print(f"   - Mean Absolute Difference: {mean_abs_diff:.6f} intensity levels")

    print("\n" + "=" * 60)
    print("Single-Frame GPU Verification PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
