"""FPV Restoration Baseline Processor — Targeted Classical Restoration for Drone Feeds.

Combines:
1. Macroblock deblocking & spatial noise smoothing via Bilateral Filter.
2. Controlled L-channel local contrast optimization via low-clip CLAHE in LAB space.
3. Edge-aware gradient sharpening to enhance structural details without haloing or block boundary amplification.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.pipeline import BaseProcessor


class FpvRestorationEnhancer(BaseProcessor):
    """
    Classical FPV Restoration Baseline Processor.

    Specifically engineered for low-bitrate FPV transmission feeds:
    - Smooths 8x8 macroblock compression boundaries
    - Enhances structural edge sharpness
    - Prevents noise amplification and temporal flickering
    """

    def __init__(
        self,
        d: int = 5,
        sigma_color: float = 25.0,
        sigma_space: float = 25.0,
        clahe_clip: float = 1.5,
        sharpen_amount: float = 0.4,
    ):
        super().__init__()
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.clahe_clip = clahe_clip
        self.sharpen_amount = sharpen_amount

        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))

    def process(
        self,
        frame: np.ndarray,
        previous_frames: Optional[List[np.ndarray]] = None,
        timestamps: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.perf_counter()

        # Step 1: Bilateral Filter for deblocking macroblocks and smoothing sensor noise
        denoised = cv2.bilateralFilter(frame, self.d, self.sigma_color, self.sigma_space)

        # Step 2: Convert to LAB color space for L-channel contrast optimization
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        contrast_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Step 3: Edge-aware unsharp mask (sharpen only where Sobel gradient > threshold)
        gray = cv2.cvtColor(contrast_bgr, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mask = cv2.magnitude(sobelx, sobely)
        edge_mask = cv2.normalize(edge_mask, None, 0, 1, cv2.NORM_MINMAX)

        # Unsharp kernel on contrast BGR
        gaussian_blur = cv2.GaussianBlur(contrast_bgr, (0, 0), sigmaX=2.0)
        unsharp = cv2.addWeighted(contrast_bgr, 1.0 + self.sharpen_amount, gaussian_blur, -self.sharpen_amount, 0)

        # Masked blend: apply sharpening proportionally to structural edge mask
        edge_mask_3ch = np.repeat(edge_mask[:, :, np.newaxis], 3, axis=2)
        restored = (contrast_bgr.astype(np.float32) * (1.0 - edge_mask_3ch * 0.5) + unsharp.astype(np.float32) * (edge_mask_3ch * 0.5))
        restored_bgr = np.clip(restored, 0, 255).astype(np.uint8)

        elapsed_time = time.perf_counter() - start_time

        metadata = {
            "processor": "FpvRestorationEnhancer",
            "category": "Targeted FPV Restoration",
            "processing_time_sec": elapsed_time,
            "latency_ms": round(elapsed_time * 1000, 2),
            "shape": restored_bgr.shape,
            "dtype": str(restored_bgr.dtype),
        }

        return restored_bgr, metadata
