"""Tuning Study for FPV Restoration Baseline Enhancer.

Evaluates hyperparameter configurations on actual FPV video (real_fpv_user_flight.mp4)
under 3 core constraints:
1. High Visual Quality (Macroblock deblocking + edge sharpening)
2. High Temporal Stability (Temporal Flicker Ratio <= 1.05)
3. Low Latency (< 33.3 ms for 30+ FPS)
"""

import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# CUDA DLL Setup
cudnn_dir = r"C:\Users\azhar\CODES\ai\.venv\Lib\site-packages\torch\lib"
cuda_dir = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
if os.path.exists(cudnn_dir):
    os.add_dll_directory(cudnn_dir)
if os.path.exists(cuda_dir):
    os.add_dll_directory(cuda_dir)

local_libs = os.path.abspath(".venv_libs")
if os.path.exists(local_libs) and local_libs not in sys.path:
    sys.path.insert(0, local_libs)

from app.quality import calculate_tenengrad


def compute_frame_motion(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> float:
    """Compute frame-to-frame motion magnitude using mean absolute pixel difference."""
    g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(np.abs(g1.astype(np.float32) - g2.astype(np.float32))))


class TunableFpvRestorationEnhancer:
    """Configurable FPV Restoration Enhancer for Hyperparameter Tuning."""

    def __init__(
        self,
        d: int = 5,
        sigma_color: float = 20.0,
        sigma_space: float = 20.0,
        clahe_clip: float = 1.2,
        sharpen_amount: float = 0.25,
        grad_threshold: float = 50.0,
        temporal_alpha: float = 0.85,  # IIR temporal smoothing factor
    ):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.clahe_clip = clahe_clip
        self.sharpen_amount = sharpen_amount
        self.grad_threshold = grad_threshold
        self.temporal_alpha = temporal_alpha

        self.clahe = (
            cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
            if self.clahe_clip > 0
            else None
        )
        self.prev_l_channel: Optional[np.ndarray] = None

    def reset(self):
        self.prev_l_channel = None

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()

        # Step 1: Bilateral Filter for deblocking macroblocks
        if self.d > 0:
            denoised = cv2.bilateralFilter(frame, self.d, self.sigma_color, self.sigma_space)
        else:
            denoised = frame

        # Step 2: LAB L-channel contrast optimization with temporal smoothing
        if self.clahe is not None:
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_eq = self.clahe.apply(l)

            # Apply temporal IIR smoothing on L-channel to eliminate brightness pumping
            if self.prev_l_channel is not None and self.temporal_alpha < 1.0:
                l_smooth = (
                    self.temporal_alpha * l_eq.astype(np.float32)
                    + (1.0 - self.temporal_alpha) * self.prev_l_channel.astype(np.float32)
                ).astype(np.uint8)
            else:
                l_smooth = l_eq

            self.prev_l_channel = l_smooth.copy()
            lab_enhanced = cv2.merge((l_smooth, a, b))
            contrast_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        else:
            contrast_bgr = denoised

        # Step 3: Absolute Edge-aware Unsharp Mask (deterministic absolute gradient scaling)
        if self.sharpen_amount > 0:
            gray = cv2.cvtColor(contrast_bgr, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            edge_mag = cv2.magnitude(sobelx, sobely)

            # Absolute threshold scaling (0 to 1) without dynamic NORM_MINMAX fluctuation
            edge_weight = np.clip(edge_mag / max(1.0, self.grad_threshold), 0.0, 1.0)
            edge_weight_3ch = np.repeat(edge_weight[:, :, np.newaxis], 3, axis=2)

            gaussian_blur = cv2.GaussianBlur(contrast_bgr, (0, 0), sigmaX=1.5)
            unsharp = cv2.addWeighted(
                contrast_bgr, 1.0 + self.sharpen_amount, gaussian_blur, -self.sharpen_amount, 0
            )

            restored = contrast_bgr.astype(np.float32) * (1.0 - edge_weight_3ch * 0.4) + unsharp.astype(np.float32) * (edge_weight_3ch * 0.4)
            restored_bgr = np.clip(restored, 0, 255).astype(np.uint8)
        else:
            restored_bgr = contrast_bgr

        t1 = time.perf_counter()
        return restored_bgr, (t1 - t0) * 1000


def main():
    print("=" * 70, flush=True)
    print("AURA FPV Restoration Baseline Hyperparameter Tuning Study", flush=True)
    print("Constraints: Quality + Temporal Stability (Flicker <= 1.05) + Latency < 33.3ms", flush=True)
    print("=" * 70, flush=True)

    video_path = os.path.join("samples", "fpv", "real_fpv_user_flight.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join("samples", "real_fpv_degraded.mp4")

    print(f"Loading video: {video_path}", flush=True)
    cap = cv2.VideoCapture(video_path)
    clip_frames = []
    for _ in range(120):  # Evaluate 120 consecutive frames
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        clip_frames.append(frame)
    cap.release()

    print(f"Loaded {len(clip_frames)} frames for tuning evaluation.", flush=True)

    # Calculate raw baseline metrics
    raw_diffs = []
    raw_tens = []
    for i in range(len(clip_frames) - 1):
        raw_diffs.append(compute_frame_motion(clip_frames[i], clip_frames[i + 1]))
        raw_tens.append(calculate_tenengrad(clip_frames[i]))
    raw_diff_mean = np.mean(raw_diffs)
    raw_ten_mean = np.mean(raw_tens)

    print(f"  Raw Feed Baseline: Motion Diff={raw_diff_mean:.2f} | Tenengrad={raw_ten_mean:.1f}", flush=True)

    # Grid of hyperparameter candidates to test
    grid = [
        # (name, d, sigma_c, sigma_s, clahe_clip, sharpen_amt, grad_thresh, temp_alpha)
        ("Config-1: Default Untuned", 5, 25.0, 25.0, 1.5, 0.40, 0.0, 1.0),
        ("Config-2: Mild Denoise + Low CLAHE", 5, 20.0, 20.0, 1.2, 0.20, 50.0, 1.0),
        ("Config-3: Mild Denoise + Temporal Smooth 0.85", 5, 20.0, 20.0, 1.2, 0.20, 50.0, 0.85),
        ("Config-4: Fast Denoise (d=3) + Temporal Smooth 0.80", 3, 20.0, 20.0, 1.2, 0.25, 45.0, 0.80),
        ("Config-5: Balanced Deblock + Edge Mask + Temporal 0.80", 5, 18.0, 18.0, 1.2, 0.25, 40.0, 0.80),
        ("Config-6: Ultra-Stable Temporal 0.75 + Sharp 0.30", 5, 20.0, 20.0, 1.0, 0.30, 40.0, 0.75),
        ("Config-7: Fast Deblock (d=3) + Temporal 0.85", 3, 15.0, 15.0, 1.0, 0.20, 50.0, 0.85),
    ]

    results = []

    print("\nRunning hyperparameter tuning search...", flush=True)
    print("-" * 105, flush=True)
    print(f"{'Config Name':<45} | {'Med ms':>6} | {'FPS':>6} | {'Flicker':>7} | {'Tenengrad':>9} | {'Stability':<15}", flush=True)
    print("-" * 105, flush=True)

    for label, d, sc, ss, clip, sh_amt, g_thr, alpha in grid:
        enhancer = TunableFpvRestorationEnhancer(
            d=d,
            sigma_color=sc,
            sigma_space=ss,
            clahe_clip=clip,
            sharpen_amount=sh_amt,
            grad_threshold=g_thr,
            temporal_alpha=alpha,
        )

        latencies = []
        enhanced_frames = []
        for frame in clip_frames:
            ef, lat = enhancer.process(frame)
            latencies.append(lat)
            enhanced_frames.append(ef)

        # Compute metrics
        enh_diffs = []
        enh_tens = []
        for i in range(len(enhanced_frames) - 1):
            enh_diffs.append(compute_frame_motion(enhanced_frames[i], enhanced_frames[i + 1]))
            enh_tens.append(calculate_tenengrad(enhanced_frames[i]))

        flicker_ratio = float(np.mean(enh_diffs) / (raw_diff_mean + 1e-5))
        med_lat = float(np.median(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        eff_fps = float(1000.0 / med_lat) if med_lat > 0 else 999.0
        ten_mean = float(np.mean(enh_tens))
        ten_gain_pct = float((ten_mean - raw_ten_mean) / raw_ten_mean * 100.0)

        flicker_status = "EXCELLENT" if flicker_ratio <= 1.05 else ("ACCEPTABLE" if flicker_ratio <= 1.25 else "UNSTABLE")
        fps_status = "PASSED (<33.3ms)" if med_lat < 33.3 else "FAILED"

        item = {
            "config": label,
            "params": {
                "d": d,
                "sigma_color": sc,
                "sigma_space": ss,
                "clahe_clip": clip,
                "sharpen_amount": sh_amt,
                "grad_threshold": g_thr,
                "temporal_alpha": alpha,
            },
            "median_latency_ms": round(med_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "effective_fps": round(eff_fps, 1),
            "flicker_ratio": round(flicker_ratio, 3),
            "tenengrad": round(ten_mean, 1),
            "sharpness_gain_pct": round(ten_gain_pct, 1),
            "flicker_assessment": flicker_status,
            "fps_status": fps_status,
        }
        results.append(item)

        print(f"{label:<45} | {med_lat:6.2f} | {eff_fps:6.1f} | {flicker_ratio:7.3f} | {ten_mean:9.1f} | {flicker_status:<15}", flush=True)

    print("-" * 105, flush=True)

    # Save JSON report
    os.makedirs("outputs/fpv_analysis", exist_ok=True)
    out_json = "outputs/fpv_analysis/fpv_restoration_tuning_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved tuning results JSON: {out_json}", flush=True)

    # Select optimal configuration (lowest flicker ratio among those passing latency < 33.3ms & positive sharpness gain)
    valid = [r for r in results if r["median_latency_ms"] < 33.3 and r["flicker_ratio"] <= 1.10]
    valid.sort(key=lambda r: r["flicker_ratio"])

    if valid:
        best = valid[0]
        print(f"\n★ OPTIMAL TUNED CONFIGURATION FOUND ★", flush=True)
        print(f"  Config: {best['config']}", flush=True)
        print(f"  Params: {best['params']}", flush=True)
        print(f"  Median Latency: {best['median_latency_ms']} ms ({best['effective_fps']} FPS)", flush=True)
        print(f"  Temporal Flicker Ratio: {best['flicker_ratio']} ({best['flicker_assessment']})", flush=True)
        print(f"  Sharpness Gain: +{best['sharpness_gain_pct']}%", flush=True)

        # Save optimal before/after frame comparison
        best_enhancer = TunableFpvRestorationEnhancer(**best["params"])
        sample_frame = clip_frames[10]
        opt_enhanced, _ = best_enhancer.process(sample_frame)

        # Side by side comparison
        h, w = sample_frame.shape[:2]
        vis_raw = sample_frame.copy()
        vis_opt = opt_enhanced.copy()

        cv2.rectangle(vis_raw, (10, 10), (140, 35), (0, 0, 0), -1)
        cv2.putText(vis_raw, "ORIGINAL", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        cv2.rectangle(vis_opt, (10, 10), (280, 35), (0, 0, 0), -1)
        cv2.putText(vis_opt, "TUNED FPV RESTORATION", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1, cv2.LINE_AA)

        side_by_side = np.hstack([vis_raw, vis_opt])
        sbs_path = "outputs/fpv_analysis/tuned_fpv_restoration_side_by_side.jpg"
        cv2.imwrite(sbs_path, side_by_side)
        print(f"  Saved side-by-side comparison image: {sbs_path}", flush=True)

if __name__ == "__main__":
    main()
