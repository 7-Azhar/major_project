"""Bilateral Filter Denoise Enhancer - Baseline 3.

Reduces sensor noise and transmission grain while preserving sharp edges.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.pipeline import BaseProcessor


class BilateralDenoiseEnhancer(BaseProcessor):
    """
    Bilateral Denoising (Baseline 3: Edge-Preserving Denoising).

    Filters noise by combining domain space gaussian weighting with range intensity weighting.
    """

    def __init__(
        self,
        d: int = 5,
        sigma_color: float = 50.0,
        sigma_space: float = 50.0,
    ):
        super().__init__()
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def process(
        self,
        frame: np.ndarray,
        previous_frames: Optional[List[np.ndarray]] = None,
        timestamps: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.perf_counter()

        enhanced_frame = cv2.bilateralFilter(
            frame, d=self.d, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space
        )

        elapsed_time = time.perf_counter() - start_time

        metadata = {
            "processor": "BilateralDenoiseEnhancer",
            "category": "Denoising",
            "d": self.d,
            "sigma_color": self.sigma_color,
            "sigma_space": self.sigma_space,
            "processing_time_sec": elapsed_time,
            "shape": enhanced_frame.shape,
            "dtype": str(enhanced_frame.dtype),
        }

        return enhanced_frame, metadata
