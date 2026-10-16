"""Objective Image Quality Assessment Module for AURA Engine.

Computes:
1. Full-Reference Metrics (when reference frame available):
   - PSNR (Peak Signal-to-Noise Ratio)
   - SSIM (Structural Similarity Index Measure)
2. No-Reference Metrics (for live feeds):
   - Tenengrad Sharpness Score (Sobel Gradient Energy)
   - Luminance Entropy / Contrast Spread
"""

import math
from typing import Dict, Any
import cv2
import numpy as np


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in dB between two images.
    Higher values indicate greater similarity to reference.
    """
    if img1.shape != img2.shape:
        return 0.0

    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")

    max_pixel = 255.0
    return 20.0 * math.log10(max_pixel / math.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two grayscale images.
    Returns float in range [-1, 1], where 1.0 indicates identical structure.
    """
    if img1.shape != img2.shape:
        return 0.0

    if len(img1.shape) == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        g1, g2 = img1, img2

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    g1 = g1.astype(np.float64)
    g2 = g2.astype(np.float64)

    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(np.mean(ssim_map))


def calculate_tenengrad(img: np.ndarray) -> float:
    """
    Compute Tenengrad Gradient Sharpness Index (No-Reference Metric).
    Measures high-frequency gradient energy using Sobel filters.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    return float(np.mean(gradient_magnitude ** 2))


def evaluate_quality_pair(raw_frame: np.ndarray, enhanced_frame: np.ndarray) -> Dict[str, float]:
    """
    Evaluate comprehensive full-reference and no-reference quality metrics.
    """
    psnr_val = calculate_psnr(raw_frame, enhanced_frame)
    ssim_val = calculate_ssim(raw_frame, enhanced_frame)
    raw_tenengrad = calculate_tenengrad(raw_frame)
    enh_tenengrad = calculate_tenengrad(enhanced_frame)

    return {
        "psnr_db": round(psnr_val, 2) if not math.isinf(psnr_val) else 999.0,
        "ssim": round(ssim_val, 4),
        "raw_tenengrad": round(raw_tenengrad, 2),
        "enhanced_tenengrad": round(enh_tenengrad, 2),
        "sharpness_gain_pct": round(((enh_tenengrad - raw_tenengrad) / max(1e-5, raw_tenengrad)) * 100.0, 2),
    }
