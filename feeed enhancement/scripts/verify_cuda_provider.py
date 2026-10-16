"""Minimal diagnostic script to test CUDAExecutionProvider initialization with explicit DLL path adding."""
import os
import sys

# Add cuDNN and CUDA DLL directories dynamically via os.add_dll_directory
cudnn_dir = r"C:\Users\azhar\CODES\ai\.venv\Lib\site-packages\torch\lib"
cuda_dir = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"

print("=" * 60)
print("Testing Explicit DLL Path Addition for CUDAExecutionProvider")
print("=" * 60)

if os.path.exists(cudnn_dir):
    os.add_dll_directory(cudnn_dir)
    print(f"Added cuDNN DLL directory: {cudnn_dir}")
else:
    print(f"cuDNN directory NOT found: {cudnn_dir}")

if os.path.exists(cuda_dir):
    os.add_dll_directory(cuda_dir)
    print(f"Added CUDA DLL directory: {cuda_dir}")
else:
    print(f"CUDA directory NOT found: {cuda_dir}")

local_libs = os.path.abspath(".venv_libs")
if os.path.exists(local_libs) and local_libs not in sys.path:
    sys.path.insert(0, local_libs)

import onnxruntime as ort

print(f"\nONNX Runtime Version: {ort.__version__}")
print(f"Available Providers: {ort.get_available_providers()}")

onnx_path = "outputs/models/zerodce.onnx"
if not os.path.exists(onnx_path):
    print(f"Error: ONNX model '{onnx_path}' not found.")
    sys.exit(1)

print("\nAttempting explicit CUDAExecutionProvider session creation (NO fallback allowed)...")

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

try:
    session = ort.InferenceSession(onnx_path, sess_options=so, providers=["CUDAExecutionProvider"])
    active_providers = session.get_providers()
    print("\n" + "=" * 60)
    print("CUDA REQUESTED: YES")
    print(f"ACTIVE PROVIDER: {active_providers}")
    if "CUDAExecutionProvider" in active_providers:
        print("CUDA PROVIDER INITIALIZATION: SUCCESS!")
    else:
        print(f"CUDA PROVIDER INITIALIZATION: FAILED (Active: {active_providers})")
    print("=" * 60)
except Exception as e:
    print("\n" + "=" * 60)
    print("CUDA REQUESTED: YES")
    print("CUDA PROVIDER INITIALIZATION: FAILED")
    print(f"EXACT ERROR: {e}")
    print("=" * 60)
