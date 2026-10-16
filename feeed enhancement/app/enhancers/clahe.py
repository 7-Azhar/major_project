"""Contrast Limited Adaptive Histogram Equalization (CLAHE) Enhancer - Baseline 1.

Enhances local illumination and shadow details in underexposed FPV footage.
Operates on the luminance channel (L in LAB color space) to prevent color distortion.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.pipeline import BaseProcessor


class ClaheEnhancer(BaseProcessor):
    """
    CLAHE (Baseline 1: Illumination & Contrast Enhancement).

    Converts BGR to LAB color space, applies CLAHE to the L channel,
    and converts back to BGR.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )

    def process(
        self,
        frame: np.ndarray,
        previous_frames: Optional[List[np.ndarray]] = None,
        timestamps: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.perf_counter()

        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to Luminance channel
        cl_channel = self.clahe.apply(l_channel)

        # Merge channels and convert back to BGR
        merged_lab = cv2.merge((cl_channel, a_channel, b_channel))
        enhanced_frame = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        elapsed_time = time.perf_counter() - start_time

        metadata = {
            "processor": "ClaheEnhancer",
            "category": "Illumination Enhancement",
            "clip_limit": self.clip_limit,
            "tile_grid_size": self.tile_grid_size,
            "processing_time_sec": elapsed_time,
            "shape": enhanced_frame.shape,
            "dtype": str(enhanced_frame.dtype),
        }

        return enhanced_frame, metadata
