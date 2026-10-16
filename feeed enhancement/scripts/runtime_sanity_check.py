"""PyTorch Runtime Sanity Check & Diagnostic Suite (Phase 3A.1)."""
import os
import sys
import time

local_libs = os.path.abspath(".venv_libs")
if os.path.exists(local_libs) and local_libs not in sys.path:
    sys.path.insert(0, local_libs)

import numpy as np
import torch
import torch.nn as nn

print("=" * 60)
print("AURA Phase 3A.1 — PyTorch Runtime Sanity Check")
print("=" * 60)

# 1. PyTorch & Environment Info
print("\n--- 1. PyTorch Build & Configuration ---")
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"PyTorch CUDA Version: {torch.version.cuda}")
print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")

print(f"MKL Built-in: {torch.backends.mkl.is_available()}")
print(f"MKLDNN Built-in: {torch.backends.mkldnn.is_available()}")
print(f"OpenMP Built-in: {torch.backends.openmp.is_available()}")

print("\n--- PyTorch Config Details ---")
config_str = torch.__config__.show()
print(config_str[:1500] if len(config_str) > 1500 else config_str)

# 2. Thread Configuration
print("\n--- 2. Threading Configuration ---")
print(f"torch.get_num_threads(): {torch.get_num_threads()}")
print(f"torch.get_num_interop_threads(): {torch.get_num_interop_threads()}")
print(f"OMP_NUM_THREADS env: {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")
print(f"MKL_NUM_THREADS env: {os.environ.get('MKL_NUM_THREADS', 'NOT SET')}")
print(f"OPENBLAS_NUM_THREADS env: {os.environ.get('OPENBLAS_NUM_THREADS', 'NOT SET')}")

# 3. Benchmark a Tiny Independent Conv2D
print("\n--- 3. Tiny Independent Conv2D Benchmark (720p, batch=1, in=3, out=32, k=3) ---")
conv = nn.Conv2d(3, 32, 3, 1, 1)
x = torch.randn(1, 3, 720, 1280)

# Warmup
with torch.no_grad():
    _ = conv(x)

t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(5):
        _ = conv(x)
t1 = time.perf_counter()
avg_single_conv_ms = ((t1 - t0) / 5.0) * 1000.0
print(f"Single Conv2d (3->32, 720p) Avg Latency: {avg_single_conv_ms:.2f} ms")

# 4. Zero-DCE Benchmark at Multiple Resolutions
from app.enhancers.neural import ZeroDceEnhancer

print("\n--- 4. Zero-DCE Resolution Scaling Diagnostic ---")
enhancer = ZeroDceEnhancer(model_path="outputs/models/Epoch99.pth", device="cpu", fp16=False, num_curves=8)

resolutions = [
    ("320x180", 180, 320),
    ("640x360", 360, 640),
    ("1280x720", 720, 1280),
]

for label, h, w in resolutions:
    dummy = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    tensor_in = enhancer.preprocess(dummy)
    t1 = time.perf_counter()
    with torch.no_grad():
        tensor_out = enhancer._forward_neural(tensor_in)
    t2 = time.perf_counter()
    _ = enhancer.postprocess(tensor_out)
    t3 = time.perf_counter()
    print(f"  Resolution {label:8s} ({h*w:7,d} px): Prep={(t1-t0)*1000:6.2f} ms | Forward={(t2-t1)*1000:8.2f} ms | Total={(t3-t0)*1000:8.2f} ms")

# 5. Thread Count Scaling Benchmark
print("\n--- 5. PyTorch Thread Count Scaling (at 640x360) ---")
original_threads = torch.get_num_threads()
dummy_640 = np.random.randint(0, 256, (360, 640, 3), dtype=np.uint8)

for t in [1, 2, 4, original_threads]:
    torch.set_num_threads(t)
    tensor_in = enhancer.preprocess(dummy_640)
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = enhancer._forward_neural(tensor_in)
    t1 = time.perf_counter()
    print(f"  Threads={t:2d}: Forward Latency = {(t1-t0)*1000:6.2f} ms")

torch.set_num_threads(original_threads)

print("\n" + "=" * 60)
print("Sanity Check Diagnostic Execution Completed Successfully")
print("=" * 60)
