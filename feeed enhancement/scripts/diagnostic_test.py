"""Single-frame diagnostic script with explicit progress markers."""
import os
import sys
import time

print("START", flush=True)

local_libs = os.path.abspath(".venv_libs")
if os.path.exists(local_libs) and local_libs not in sys.path:
    sys.path.insert(0, local_libs)

import numpy as np
import torch
from app.enhancers.neural import ZeroDceEnhancer

print("IMPORTS COMPLETE", flush=True)
print(f"Python version: {sys.version.split()[0]}", flush=True)
print(f"PyTorch version: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
print(f"PyTorch num_threads: {torch.get_num_threads()}", flush=True)
print(f"PyTorch num_interop_threads: {torch.get_num_interop_threads()}", flush=True)

print("MODEL INITIALIZATION START", flush=True)
t_init0 = time.perf_counter()
enhancer = ZeroDceEnhancer(model_path="outputs/models/Epoch99.pth", device="cpu", fp16=False, num_curves=8)
t_init1 = time.perf_counter()
print("MODEL INITIALIZATION COMPLETE", flush=True)
print(f"Model initialization time: {(t_init1 - t_init0)*1000:.2f} ms", flush=True)

if enhancer.net is not None:
    total_params = sum(p.numel() for p in enhancer.net.parameters())
    print(f"Model parameter count: {total_params:,}", flush=True)
else:
    print("Model parameter count: N/A (net is None)", flush=True)

np.random.seed(42)
frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

print("PREPROCESS START", flush=True)
t0 = time.perf_counter()
tensor_in = enhancer.preprocess(frame)
t1 = time.perf_counter()
print("PREPROCESS COMPLETE", flush=True)
print(f"Preprocess time: {(t1 - t0)*1000:.2f} ms", flush=True)

print("FORWARD START", flush=True)
with torch.no_grad():
    tensor_out = enhancer._forward_neural(tensor_in)
t2 = time.perf_counter()
print("FORWARD COMPLETE", flush=True)
print(f"Forward time: {(t2 - t1)*1000:.2f} ms", flush=True)

print("POSTPROCESS START", flush=True)
out_frame = enhancer.postprocess(tensor_out)
t3 = time.perf_counter()
print("POSTPROCESS COMPLETE", flush=True)
print(f"Postprocess time: {(t3 - t2)*1000:.2f} ms", flush=True)

print("TOTAL COMPLETE", flush=True)
print(f"Total time: {(t3 - t0)*1000:.2f} ms", flush=True)
