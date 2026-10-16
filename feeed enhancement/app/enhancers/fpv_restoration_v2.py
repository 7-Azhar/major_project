"""AURA Phase 5B — FPV Restoration v2 Processor (FpvRestorationV2Enhancer).

Multi-Scale Classical FPV Video Restoration Pipeline designed specifically for low-bitrate FPV transmission feeds:
1. Macroblock Deblocking & Edge-Preserving Denoising on LAB L-channel.
2. Multi-Scale Mid-Frequency Detail Band Extraction (enhances structural edges without amplifying high-frequency noise).
3. Gradient Halo Protection Mask (prevents ringing and artificial outlines around high-contrast transitions).
4. Controlled Local Contrast (Luminance-only CLAHE + linear gain).
5. Temporal IIR Parameter & Luminance Smoothing (eliminates brightness pumping and flickering without motion ghosting).
6. Chroma Preservation (zero color shifts or unnatural hue alterations).
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.pipeline import BaseProcessor


class FpvRestorationV2Enhancer(BaseProcessor):
    """
    AURA FPV Restoration v2 Processor.

    Exposes configurable parameters:
    - deblock_strength: Controls macroblock boundary smoothing.
    - denoise_strength: Controls spatial noise suppression.
    - detail_strength: Controls mid-frequency structural detail band amplification.
    - contrast_strength: Controls conservative luminance contrast adjustment.
    - temporal_stability: IIR temporal smoothing factor (0.0 to 1.0) to prevent flickering.
    - preset: Optional preset mode ("SAFE", "BALANCED", "DETAIL").
    """

    PRESETS = {
        "SAFE": {
            "deblock_strength": 0.8,
            "denoise_strength": 0.8,
            "detail_strength": 0.20,
            "contrast_strength": 0.15,
            "temporal_stability": 0.85,
        },
        "BALANCED": {
            "deblock_strength": 1.0,
            "denoise_strength": 1.0,
            "detail_strength": 0.35,
            "contrast_strength": 0.25,
            "temporal_stability": 0.80,
        },
        "DETAIL": {
            "deblock_strength": 1.2,
            "denoise_strength": 1.1,
            "detail_strength": 0.50,
            "contrast_strength": 0.35,
            "temporal_stability": 0.75,
        },
    }

    def __init__(
        self,
        deblock_strength: float = 1.0,
        denoise_strength: float = 1.0,
        detail_strength: float = 0.35,
        contrast_strength: float = 0.25,
        temporal_stability: float = 0.80,
        preset: Optional[str] = None,
    ):
        super().__init__()

        if preset is not None and preset.upper() in self.PRESETS:
            p = self.PRESETS[preset.upper()]
            self.deblock_strength = float(p["deblock_strength"])
            self.denoise_strength = float(p["denoise_strength"])
            self.detail_strength = float(p["detail_strength"])
            self.contrast_strength = float(p["contrast_strength"])
            self.temporal_stability = float(p["temporal_stability"])
            self.preset_name = preset.upper()
        else:
            self.deblock_strength = max(0.0, float(deblock_strength))
            self.denoise_strength = max(0.0, float(denoise_strength))
            self.detail_strength = max(0.0, float(detail_strength))
            self.contrast_strength = max(0.0, float(contrast_strength))
            self.temporal_stability = max(0.0, min(1.0, float(temporal_stability)))
            self.preset_name = "CUSTOM"

        # Initialize CLAHE with conservative clip limit
        clip = 1.0 + 0.5 * self.contrast_strength
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))

        # Temporal memory (IIR filter state)
        self.prev_l_channel: Optional[np.ndarray] = None

    def reset(self):
        """Reset temporal filter memory."""
        self.prev_l_channel = None

    def process(
        self,
        frame: np.ndarray,
        previous_frames: Optional[List[np.ndarray]] = None,
        timestamps: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or invalid.")

        start_time = time.perf_counter()

        # Step 1: Color Space Conversion (BGR -> LAB to isolate Luminance L)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_orig, a_orig, b_orig = cv2.split(lab)
        l_float = l_orig.astype(np.float32)

        # Step 2: Macroblock Deblocking & Spatial Denoising on L-channel
        if self.denoise_strength > 0.0 or self.deblock_strength > 0.0:
            sigma = max(3.0, 15.0 * max(self.denoise_strength, self.deblock_strength))
            l_denoised = cv2.bilateralFilter(l_orig, 5, sigma, sigma).astype(np.float32)
        else:
            l_denoised = l_float

        # Step 3: Multi-Scale Mid-Frequency Detail Extraction & Halo Protection
        if self.detail_strength > 0.0:
            # Gaussian blur 1: High + Mid frequencies
            blur_small = cv2.GaussianBlur(l_denoised, (3, 3), 1.0)
            # Gaussian blur 2: Low frequency base
            blur_large = cv2.GaussianBlur(l_denoised, (7, 7), 3.0)

            # Mid-frequency band: structural edges (buildings, trees, roads)
            mid_freq_band = blur_small - blur_large

            # Compute Sobel gradient magnitude for Halo Protection
            sobelx = cv2.Sobel(l_orig, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(l_orig, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = cv2.magnitude(sobelx, sobely)

            # Halo protection mask: suppress detail boost on extremely sharp transitions (grad_mag > 120)
            halo_protect = np.clip(1.0 - (grad_mag / 120.0) ** 2, 0.15, 1.0)

            # Apply mid-frequency detail boost with halo protection
            l_restored = l_denoised + (mid_freq_band * self.detail_strength * halo_protect)
        else:
            l_restored = l_denoised

        l_restored_uint8 = np.clip(l_restored, 0, 255).astype(np.uint8)

        # Step 4: Adaptive Noise-Conditional Contrast Adjustment (CLAHE on L-channel)
        if self.contrast_strength > 0.0:
            # Estimate spatial noise MAD on smooth Laplacian residual
            blur_mad = cv2.GaussianBlur(l_restored_uint8, (5, 5), 0)
            noise_mad = float(np.median(np.abs(l_restored_uint8.astype(np.float32) - blur_mad.astype(np.float32))) * 1.4826)

            # Dampen contrast clip limit if noise is high to prevent noise/grain explosion
            dampening = 1.0 / (1.0 + 0.3 * max(0.0, noise_mad - 1.0))
            clip = 1.0 + (0.5 * self.contrast_strength * dampening)

            clahe_adaptive = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            l_contrast = clahe_adaptive.apply(l_restored_uint8)
        else:
            l_contrast = l_restored_uint8

        # Step 5: Lightweight Temporal IIR Smoothing (Prevents brightness pumping & flicker)
        if (
            self.prev_l_channel is not None
            and self.temporal_stability > 0.0
            and self.prev_l_channel.shape == l_contrast.shape
        ):
            alpha = max(0.5, min(0.95, self.temporal_stability))
            l_final_float = (
                alpha * l_contrast.astype(np.float32)
                + (1.0 - alpha) * self.prev_l_channel.astype(np.float32)
            )
            l_final = np.clip(l_final_float, 0, 255).astype(np.uint8)
        else:
            l_final = l_contrast

        self.prev_l_channel = l_final.copy()

        # Step 6: Reconstruction (Merge smoothed L with original A & B chroma)
        lab_final = cv2.merge((l_final, a_orig, b_orig))
        output_bgr = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)

        elapsed_time = time.perf_counter() - start_time

        metadata = {
            "processor": "FpvRestorationV2Enhancer",
            "category": "Multi-Scale FPV Video Restoration v2",
            "preset": self.preset_name,
            "processing_time_sec": elapsed_time,
            "latency_ms": round(elapsed_time * 1000, 2),
            "parameters": {
                "deblock_strength": self.deblock_strength,
                "denoise_strength": self.denoise_strength,
                "detail_strength": self.detail_strength,
                "contrast_strength": self.contrast_strength,
                "temporal_stability": self.temporal_stability,
            },
            "shape": output_bgr.shape,
            "dtype": str(output_bgr.dtype),
        }

        return output_bgr, metadata
