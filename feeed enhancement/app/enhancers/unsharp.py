"""Unsharp Mask Enhancer - Baseline 2.

Boosts high-frequency edge contrast to sharpen motion-blurred or soft FPV footage.
Note: Does not recover lost information; boosts high-frequency edge gradients.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.pipeline import BaseProcessor


class UnsharpMaskEnhancer(BaseProcessor):
    """
    Unsharp Masking (Baseline 2: Sharpening & Edge Enhancement).

    Applies Gaussian blur and subtracts it from the original to amplify edge gradients.
    formula: enhanced = original + amount * (original - blurred)
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (5, 5),
        sigma: float = 1.0,
        amount: float = 1.5,
        threshold: int = 0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.amount = amount
        self.threshold = threshold

    def process(
        self,
        frame: np.ndarray,
        previous_frames: Optional[List[np.ndarray]] = None,
        timestamps: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.perf_counter()

        blurred = cv2.GaussianBlur(frame, self.kernel_size, self.sigma)

        if self.threshold > 0:
            low_contrast_mask = cv2.absdiff(frame, blurred) < self.threshold
            sharpened = cv2.addWeighted(frame, 1.0 + self.amount, blurred, -self.amount, 0)
            enhanced_frame = np.where(low_contrast_mask, frame, sharpened)
        else:
            enhanced_frame = cv2.addWeighted(frame, 1.0 + self.amount, blurred, -self.amount, 0)

        elapsed_time = time.perf_counter() - start_time

        metadata = {
            "processor": "UnsharpMaskEnhancer",
            "category": "Sharpening",
            "kernel_size": self.kernel_size,
            "sigma": self.sigma,
            "amount": self.amount,
            "threshold": self.threshold,
            "processing_time_sec": elapsed_time,
            "shape": enhanced_frame.shape,
            "dtype": str(enhanced_frame.dtype),
        }

        return enhanced_frame, metadata
