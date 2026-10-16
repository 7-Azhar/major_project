"""AURA Phase 5A — Hybrid FPV Restoration Prototype (FpvHybridEnhancer).

Combines:
1. Bilateral Denoise -> fast macroblock boundary & sensor noise suppression.
2. FPV Restoration Core -> LAB luminance deblocking & soft local contrast.
3. Controlled Luminance Contrast -> conservative visibility adjustment without brightness pumping.
4. Mild Edge Detail Recovery -> edge-gradient-masked detail sharpening without haloing.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.pipeline import BaseProcessor


class FpvHybridEnhancer(BaseProcessor):
    """
    Experimental Hybrid FPV Restoration Processor.

    Exposes configurable parameters:
    - bilateral_strength: Controls bilateral filter noise/deblocking intensity.
    - restoration_strength: Controls restoration core blending.
    - contrast_strength: Controls luminance-only contrast adjustment (conservative).
    - detail_strength: Controls edge-masked detail sharpening without haloing.
    """

    def __init__(
        self,
        bilateral_strength: float = 1.0,
        restoration_strength: float = 1.0,
        contrast_strength: float = 0.2,
        detail_strength: float = 0.2,
    ):
        super().__init__()
        self.bilateral_strength = max(0.0, float(bilateral_strength))
        self.restoration_strength = max(0.0, float(restoration_strength))
        self.contrast_strength = max(0.0, float(contrast_strength))
        self.detail_strength = max(0.0, float(detail_strength))

        # CLAHE with conservative clip limit
        clip = 1.0 + 0.5 * self.contrast_strength
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))

    def process(
        self,
        frame: np.ndarray,
        previous_frames: Optional[List[np.ndarray]] = None,
        timestamps: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or invalid.")

        start_time = time.perf_counter()

        # Step 1: Bilateral Denoise for macroblock & high-frequency sensor noise
        if self.bilateral_strength > 0.0:
            sigma = max(5.0, 15.0 * self.bilateral_strength)
            denoised = cv2.bilateralFilter(frame, 5, sigma, sigma)
        else:
            denoised = frame

        # Step 2: FPV Restoration Core — LAB space L-channel contrast
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)

        # Step 3: Controlled Luminance/Contrast Adjustment
        if self.contrast_strength > 0.0:
            # Conservative linear stretch on L-channel
            gain = 1.0 + (0.1 * self.contrast_strength)
            l_adjusted = cv2.multiply(l_enhanced, gain)
            l_enhanced = np.clip(l_adjusted, 0, 255).astype(np.uint8)

        lab_enhanced = cv2.merge((l_enhanced, a, b))
        restored_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Blend restoration core with denoised frame based on restoration_strength
        if self.restoration_strength < 1.0:
            restored_bgr = cv2.addWeighted(
                restored_bgr,
                self.restoration_strength,
                denoised,
                1.0 - self.restoration_strength,
                0,
            )

        # Step 4: Mild Edge-Masked Detail Recovery (no halos or ringing)
        if self.detail_strength > 0.0:
            gray = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            edge_mag = cv2.magnitude(sobelx, sobely)
            edge_mask = np.clip(edge_mag / 50.0, 0.0, 1.0)
            edge_mask_3ch = np.repeat(edge_mask[:, :, np.newaxis], 3, axis=2)

            gaussian_blur = cv2.GaussianBlur(restored_bgr, (0, 0), sigmaX=1.5)
            unsharp = cv2.addWeighted(
                restored_bgr,
                1.0 + self.detail_strength,
                gaussian_blur,
                -self.detail_strength,
                0,
            )

            # Blend unsharp sharpening ONLY on edge boundaries
            final_float = restored_bgr.astype(np.float32) * (
                1.0 - edge_mask_3ch * 0.3
            ) + unsharp.astype(np.float32) * (edge_mask_3ch * 0.3)
            output_bgr = np.clip(final_float, 0, 255).astype(np.uint8)
        else:
            output_bgr = restored_bgr

        elapsed_time = time.perf_counter() - start_time

        metadata = {
            "processor": "FpvHybridEnhancer",
            "category": "Hybrid Classical FPV Restoration",
            "processing_time_sec": elapsed_time,
            "latency_ms": round(elapsed_time * 1000, 2),
            "parameters": {
                "bilateral_strength": self.bilateral_strength,
                "restoration_strength": self.restoration_strength,
                "contrast_strength": self.contrast_strength,
                "detail_strength": self.detail_strength,
            },
            "shape": output_bgr.shape,
            "dtype": str(output_bgr.dtype),
        }

        return output_bgr, metadata
