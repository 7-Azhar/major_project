"""Hardware and Runtime Detection for AURA Phase 3A."""
import os
import platform
import sys

# Ensure local libs directory is in path
local_libs = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv_libs"))
if os.path.exists(local_libs) and local_libs not in sys.path:
    sys.path.insert(0, local_libs)

import numpy as np
import cv2

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    onnx = None

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    ort = None

print("=" * 60)
print("AURA Phase 3A — Hardware & Runtime Environment Detection")
print("=" * 60)

print(f"\nPython: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
print(f"Processor: {platform.processor()}")

# System Memory
try:
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"System RAM: {ram_gb:.2f} GB")
except Exception:
    print("System RAM: N/A")

# PyTorch
print("\n--- PyTorch ---")
if HAS_TORCH:
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Device 0: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch: NOT INSTALLED")

# ONNX & ONNX Runtime
print("\n--- ONNX & ONNX Runtime ---")
if HAS_ONNX:
    print(f"ONNX Version: {onnx.__version__}")
else:
    print("ONNX: NOT INSTALLED")

if HAS_ORT:
    print(f"ONNX Runtime Version: {ort.__version__}")
    providers = ort.get_available_providers()
    print("\nONNX Runtime Providers:")
    for p in ["CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider", "DmlExecutionProvider"]:
        status = "AVAILABLE" if p in providers else "UNAVAILABLE"
        print(f"  - {p}: {status}")
else:
    print("ONNX Runtime: NOT INSTALLED")

print("\n" + "=" * 60)
