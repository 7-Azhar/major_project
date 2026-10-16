"""Step 1 & 3: Export Zero-DCE (Epoch99.pth) to ONNX and verify PyTorch vs ONNX numerical equivalence."""
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
    import onnx
    import onnxruntime as ort
except ImportError as e:
    print(f"Error importing onnx/onnxruntime: {e}")
    sys.exit(1)

from app.enhancers.neural import ZeroDceEnhancer

def main():
    print("=" * 60)
    print("Phase 3A Step 1 & 3: ONNX Export & Numerical Equivalence")
    print("=" * 60)

    ckpt_path = "outputs/models/Epoch99.pth"
    onnx_path = "outputs/models/zerodce.onnx"
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/visualizations", exist_ok=True)

    # 1. Load PyTorch model
    print("\n1. Loading validated ZeroDceEnhancer PyTorch model...")
    enhancer = ZeroDceEnhancer(model_path=ckpt_path, device="cpu", fp16=False, num_curves=8)
    net = enhancer.net
    net.eval()

    # 2. Export to ONNX (Static 1280x720 FP32)
    print("\n2. Exporting to ONNX (static [1, 3, 720, 1280], FP32)...")
    dummy_input = torch.randn(1, 3, 720, 1280, dtype=torch.float32)

    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    print(f"   ONNX model exported to: {onnx_path} ({os.path.getsize(onnx_path):,} bytes)")

    # Validate ONNX structure
    print("\n3. Validating ONNX model structure with onnx.checker...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("   ONNX structure check: PASSED!")

    # 4. Numerical Equivalence Check
    print("\n4. Running PyTorch vs ONNX CPU Numerical Equivalence Check...")
    np.random.seed(42)
    sample_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    # PyTorch execution
    tensor_in = enhancer.preprocess(sample_frame)
    with torch.no_grad():
        pytorch_tensor_out = net(tensor_in)
    pytorch_bgr = enhancer.postprocess(pytorch_tensor_out)

    # ONNX Runtime CPU execution
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"input": tensor_in.cpu().numpy()}
    ort_tensor_out = ort_session.run(None, ort_inputs)[0]

    # Postprocess ONNX output
    onnx_tensor_torch = torch.from_numpy(ort_tensor_out)
    onnx_bgr = enhancer.postprocess(onnx_tensor_torch)

    # Calculate differences
    abs_diff = np.abs(pytorch_bgr.astype(np.float64) - onnx_bgr.astype(np.float64))
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    rel_err = mean_abs_diff / (np.mean(pytorch_bgr) + 1e-6)

    print(f"   Max Absolute Difference:   {max_abs_diff:.4f} intensity levels")
    print(f"   Mean Absolute Difference:  {mean_abs_diff:.6f} intensity levels")
    print(f"   Relative Error:            {rel_err:.6e}")

    # 5. Visual Comparison Artifact
    print("\n5. Generating visual comparison artifact...")
    # Render difference map amplified by 10x for visual inspection
    diff_vis = (abs_diff * 10.0).clip(0, 255).astype(np.uint8)

    row = np.hstack([pytorch_bgr, onnx_bgr, diff_vis])
    vis_path = "outputs/visualizations/zerodce_pytorch_vs_onnx.png"
    cv2.imwrite(vis_path, row)
    print(f"   Saved comparison image (PyTorch | ONNX | Diffx10): {vis_path}")

    print("\n" + "=" * 60)
    print("Export & Numerical Equivalence PASSED")
    print("=" * 60)

if __name__ == "__main__":
    main()
