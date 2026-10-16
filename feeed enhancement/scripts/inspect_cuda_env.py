"""Inspect ONNX Runtime CUDA dependencies, system DLLs, and PATH."""
import os
import sys

local_libs = os.path.abspath(".venv_libs")
if os.path.exists(local_libs) and local_libs not in sys.path:
    sys.path.insert(0, local_libs)

import onnxruntime as ort

print("=" * 60)
print("AURA Phase 3A.2 — ONNX Runtime CUDA Environment Inspection")
print("=" * 60)

print(f"\n1. ONNX Runtime Version: {ort.__version__}")
print(f"   Import Path: {ort.__file__}")

print("\n2. Requirements based on ONNX Runtime 1.29.0 GPU documentation:")
print("   - Expected CUDA major: 13.* (cublasLt64_13.dll)")
print("   - Expected cuDNN major: 9.* (cudnn64_9.dll)")
print("   - Expected MSVC runtime: Latest Visual C++ Redistributable")

print("\n3. Inspecting System PATH for CUDA/cuDNN DLLs:")
path_dirs = os.environ.get("PATH", "").split(os.path.pathsep)
cuda_related_paths = [p for p in path_dirs if "cuda" in p.lower() or "nvidia" in p.lower() or "cudnn" in p.lower()]

if cuda_related_paths:
    for p in cuda_related_paths:
        print(f"   - {p}")
else:
    print("   No CUDA/cuDNN directories found in system PATH!")

# Search common NVIDIA/CUDA system directories for DLLs
common_search_dirs = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    r"C:\Program Files\NVIDIA Corporation",
    r"C:\tools",
    r"C:\cuda",
]

found_dlls = []
for base in common_search_dirs:
    if os.path.exists(base):
        for root, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith(".dll") and ("cuda" in f.lower() or "cublas" in f.lower() or "cudnn" in f.lower()):
                    found_dlls.append(os.path.join(root, f))

print("\n4. Searching standard Windows directories for CUDA/cuDNN DLLs:")
if found_dlls:
    for dll in found_dlls[:10]:
        print(f"   - {dll}")
    if len(found_dlls) > 10:
        print(f"   ... and {len(found_dlls) - 10} more DLLs.")
else:
    print("   No system CUDA/cuDNN toolkit or DLLs installed in standard system paths.")

print("\n" + "=" * 60)
