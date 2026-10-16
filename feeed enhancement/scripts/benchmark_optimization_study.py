"""Phase 3A.3 — Zero-DCE FP16, Curve Iterations, and Resolution Study."""
import csv
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

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import onnx
    import onnxruntime as ort
except ImportError as e:
    print(f"Error importing onnx/onnxruntime: {e}")
    sys.exit(1)

from app.enhancers.neural import ZeroDCENet, ZeroDceEnhancer
from app.quality import calculate_tenengrad

# Configurable Curve Iterations Model for PyTorch / ONNX Export
class ConfigurableZeroDCENet(nn.Module):
    def __init__(self, base_net: ZeroDCENet, num_curves: int = 8):
        super().__init__()
        self.num_curves = num_curves
        self.conv1 = base_net.conv1
        self.conv2 = base_net.conv2
        self.conv3 = base_net.conv3
        self.conv4 = base_net.conv4
        self.conv5 = base_net.conv5
        self.conv6 = base_net.conv6
        self.conv7 = base_net.conv7
        self.relu = base_net.relu
        self.tanh = base_net.tanh

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))

        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        a = self.tanh(self.conv7(torch.cat([x1, x6], dim=1)))

        curr = x
        for i in range(min(self.num_curves, 8)):
            a_i = a[:, i * 3 : (i + 1) * 3, :, :]
            curr = curr + a_i * (torch.pow(curr, 2) - curr)
        return curr

def export_variant_onnx(base_net: ZeroDCENet, num_curves: int, fp16: bool, out_path: str, height: int = 720, width: int = 1280):
    model = ConfigurableZeroDCENet(base_net, num_curves=num_curves)
    model.eval()

    if fp16:
        model = model.half()
        dummy_in = torch.randn(1, 3, height, width, dtype=torch.float16)
    else:
        dummy_in = torch.randn(1, 3, height, width, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_in,
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    return out_path

def benchmark_onnx_variant(onnx_path: str, fp16: bool, target_h: int = 720, target_w: int = 1280, warmup: int = 5, timed: int = 20):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options=so, providers=["CUDAExecutionProvider"])

    # CRITICAL: Verify CUDA provider is actually active
    active = session.get_providers()
    if "CUDAExecutionProvider" not in active:
        print(f"  FATAL: CUDAExecutionProvider NOT active! Active: {active}", flush=True)
        print(f"  Aborting to prevent CPU fallback (~55s/frame).", flush=True)
        sys.exit(1)
    print(f"  Provider verified: {active[0]}", flush=True)

    enhancer = ZeroDceEnhancer(model_path="outputs/models/Epoch99.pth", device="cpu", fp16=False, num_curves=8)
    np.random.seed(42)
    sample_720p = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup):
        if target_h != 720 or target_w != 1280:
            frame_resized = cv2.resize(sample_720p, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_resized = sample_720p

        rgb = frame_resized[:, :, ::-1].astype(np.float16 if fp16 else np.float32) / 255.0
        tensor_in = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).numpy()
        _ = session.run(None, {"input": tensor_in})[0]

    prep_times, infer_times, post_times, tot_times = [], [], [], []
    last_output_bgr = None

    for _ in range(timed):
        t0 = time.perf_counter()
        if target_h != 720 or target_w != 1280:
            frame_resized = cv2.resize(sample_720p, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_resized = sample_720p

        rgb = frame_resized[:, :, ::-1].astype(np.float16 if fp16 else np.float32) / 255.0
        tensor_in = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).numpy()
        t1 = time.perf_counter()

        ort_out = session.run(None, {"input": tensor_in})[0]
        t2 = time.perf_counter()

        out_tensor = torch.from_numpy(ort_out).float().squeeze(0).clamp(0.0, 1.0).numpy()
        out_rgb = (out_tensor.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        out_bgr = out_rgb[:, :, ::-1]

        if target_h != 720 or target_w != 1280:
            out_bgr = cv2.resize(out_bgr, (1280, 720), interpolation=cv2.INTER_LINEAR)
        t3 = time.perf_counter()

        prep_times.append((t1 - t0) * 1000)
        infer_times.append((t2 - t1) * 1000)
        post_times.append((t3 - t2) * 1000)
        tot_times.append((t3 - t0) * 1000)
        last_output_bgr = out_bgr

    tot_arr = np.array(tot_times)
    infer_arr = np.array(infer_times)

    return {
        "gpu_infer_median": float(np.median(infer_arr)),
        "total_median": float(np.median(tot_arr)),
        "total_p95": float(np.percentile(tot_arr, 95)),
        "fps": float(1000.0 / np.mean(tot_arr)),
        "output_bgr": last_output_bgr,
    }

def main():
    print("=" * 60, flush=True)
    print("AURA Phase 3A.3 — Zero-DCE Optimization & Trade-Off Study", flush=True)
    print("=" * 60, flush=True)

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/visualizations", exist_ok=True)

    base_enhancer = ZeroDceEnhancer(model_path="outputs/models/Epoch99.pth", device="cpu", fp16=False, num_curves=8)
    base_net = base_enhancer.net

    # Reference 720p input
    np.random.seed(42)
    sample_720p = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    # Reference PyTorch CPU output & FP32 8-curve ONNX output
    print("\n[1/5] Running FP32 reference benchmark...", flush=True)
    ref_fp32_onnx = "outputs/models/zerodce.onnx"
    ref_bench = benchmark_onnx_variant(ref_fp32_onnx, fp16=False, target_h=720, target_w=1280)
    ref_bgr = ref_bench["output_bgr"]
    print(f"  FP32 reference done: {ref_bench['total_median']:.2f} ms, {ref_bench['fps']:.2f} FPS", flush=True)

    results = []

    # Part A: FP16 Benchmark (8 curves, 720p)
    print("\n[2/5] --- Part A: FP16 Benchmark (8 Curves, 720p) ---", flush=True)
    fp16_onnx_path = "outputs/models/zerodce_fp16.onnx"
    print("  Exporting FP16 ONNX...", flush=True)
    export_variant_onnx(base_net, num_curves=8, fp16=True, out_path=fp16_onnx_path, height=720, width=1280)
    print("  Benchmarking FP16...", flush=True)
    bench_fp16 = benchmark_onnx_variant(fp16_onnx_path, fp16=True, target_h=720, target_w=1280)

    abs_diff_fp16 = np.abs(bench_fp16["output_bgr"].astype(np.float64) - ref_bgr.astype(np.float64))
    max_diff_fp16 = np.max(abs_diff_fp16)
    mean_diff_fp16 = np.mean(abs_diff_fp16)

    print(f"  FP32 Total Latency: {ref_bench['total_median']:.2f} ms | FPS: {ref_bench['fps']:.2f}", flush=True)
    print(f"  FP16 Total Latency: {bench_fp16['total_median']:.2f} ms | FPS: {bench_fp16['fps']:.2f}", flush=True)
    print(f"  FP16 Max Abs Diff:  {max_diff_fp16:.2f} | Mean Abs Diff: {mean_diff_fp16:.6f}", flush=True)

    results.append({
        "config": "Zero-DCE FP32 (8 Curves, 720p)",
        "precision": "FP32",
        "curves": 8,
        "resolution": "1280x720",
        "gpu_infer_ms": ref_bench["gpu_infer_median"],
        "total_ms": ref_bench["total_median"],
        "fps": ref_bench["fps"],
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "tenengrad": float(calculate_tenengrad(ref_bgr)),
    })

    results.append({
        "config": "Zero-DCE FP16 (8 Curves, 720p)",
        "precision": "FP16",
        "curves": 8,
        "resolution": "1280x720",
        "gpu_infer_ms": bench_fp16["gpu_infer_median"],
        "total_ms": bench_fp16["total_median"],
        "fps": bench_fp16["fps"],
        "max_abs_diff": float(max_diff_fp16),
        "mean_abs_diff": float(mean_diff_fp16),
        "tenengrad": float(calculate_tenengrad(bench_fp16["output_bgr"])),
    })

    # Part B: Curve Iteration Study (8, 6, 4, 2 curves in FP16)
    print("\n[3/5] --- Part B: Curve Iteration Study (FP16, 720p) ---", flush=True)
    for curves in [6, 4, 2]:
        print(f"  Exporting/benchmarking {curves}-curve variant...", flush=True)
        onnx_curve_path = f"outputs/models/zerodce_fp16_c{curves}.onnx"
        export_variant_onnx(base_net, num_curves=curves, fp16=True, out_path=onnx_curve_path, height=720, width=1280)
        b_res = benchmark_onnx_variant(onnx_curve_path, fp16=True, target_h=720, target_w=1280)

        diff = np.abs(b_res["output_bgr"].astype(np.float64) - ref_bgr.astype(np.float64))
        print(f"  Curves={curves:1d} | GPU Infer: {b_res['gpu_infer_median']:6.2f} ms | Total: {b_res['total_median']:6.2f} ms | FPS: {b_res['fps']:5.2f} | Mean Diff: {np.mean(diff):.4f}", flush=True)

        results.append({
            "config": f"Zero-DCE FP16 ({curves} Curves, 720p)",
            "precision": "FP16",
            "curves": curves,
            "resolution": "1280x720",
            "gpu_infer_ms": b_res["gpu_infer_median"],
            "total_ms": b_res["total_median"],
            "fps": b_res["fps"],
            "max_abs_diff": float(np.max(diff)),
            "mean_abs_diff": float(np.mean(diff)),
            "tenengrad": float(calculate_tenengrad(b_res["output_bgr"])),
        })

    # Part C: Resolution Study (960x540 and 640x360 in FP16, 4 curves)
    print("\n[4/5] --- Part C: Resolution Reduction Study (FP16, 4 Curves) ---", flush=True)
    res_list = [
        ("960x540", 540, 960),
        ("640x360", 360, 640),
    ]

    for label, h, w in res_list:
        print(f"  Exporting/benchmarking {label} variant...", flush=True)
        onnx_res_path = f"outputs/models/zerodce_fp16_c4_{label}.onnx"
        export_variant_onnx(base_net, num_curves=4, fp16=True, out_path=onnx_res_path, height=h, width=w)
        b_res = benchmark_onnx_variant(onnx_res_path, fp16=True, target_h=h, target_w=w)

        diff = np.abs(b_res["output_bgr"].astype(np.float64) - ref_bgr.astype(np.float64))
        print(f"  Res {label:7s} | GPU Infer: {b_res['gpu_infer_median']:6.2f} ms | Total: {b_res['total_median']:6.2f} ms | FPS: {b_res['fps']:5.2f} | Mean Diff: {np.mean(diff):.4f}", flush=True)

        results.append({
            "config": f"Zero-DCE FP16 (4 Curves, {label})",
            "precision": "FP16",
            "curves": 4,
            "resolution": label,
            "gpu_infer_ms": b_res["gpu_infer_median"],
            "total_ms": b_res["total_median"],
            "fps": b_res["fps"],
            "max_abs_diff": float(np.max(diff)),
            "mean_abs_diff": float(np.mean(diff)),
            "tenengrad": float(calculate_tenengrad(b_res["output_bgr"])),
        })

    # Save JSON & CSV Reports
    os.makedirs("outputs", exist_ok=True)
    json_path = "outputs/phase3_curve_iteration_benchmark.json"
    csv_path = "outputs/phase3_curve_iteration_benchmark.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved JSON report: {json_path}", flush=True)
    print(f"Saved CSV report:  {csv_path}", flush=True)

    # Summary Table
    print("\n[5/5] --- Summary ---", flush=True)
    print("=" * 85, flush=True)
    print(f"{'Configuration':<35} | {'GPU (ms)':<9} | {'Total (ms)':<10} | {'FPS':<6} | {'30 FPS Budget':<14}", flush=True)
    print("=" * 85, flush=True)
    for r in results:
        status = "ACHIEVED" if r["total_ms"] <= 33.33 else f"Exceeds by +{r['total_ms'] - 33.33:.1f}ms"
        print(f"{r['config']:<35} | {r['gpu_infer_ms']:9.2f} | {r['total_ms']:10.2f} | {r['fps']:6.2f} | {status:<14}", flush=True)
    print("=" * 85, flush=True)
    print("\nBenchmark study complete!", flush=True)

if __name__ == "__main__":
    main()
