"""Composite Classical Enhancer - Baseline 4.

Combines Denoising -> CLAHE Contrast -> Unsharp Masking in a strictly controlled order
to prevent noise amplification.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.enhancers.bilateral import BilateralDenoiseEnhancer
from app.enhancers.clahe import ClaheEnhancer
from app.enhancers.unsharp import UnsharpMaskEnhancer
from app.pipeline import BaseProcessor


class CompositeEnhancer(BaseProcessor):
    """
    Composite Classical Pipeline (Baseline 4).

    Executes a 3-stage controlled pipeline:
    1. Bilateral Denoising (suppress noise first)
    2. CLAHE (equalize local illumination)
    3. Unsharp Masking (restore edge sharpness)
    """

    def __init__(
        self,
        denoiser: Optional[BaseProcessor] = None,
        contrast_enhancer: Optional[BaseProcessor] = None,
        sharpener: Optional[BaseProcessor] = None,
    ):
        super().__init__()
        self.denoiser = denoiser if denoiser is not None else BilateralDenoiseEnhancer(d=3, sigma_color=30.0, sigma_space=30.0)
        self.contrast_enhancer = contrast_enhancer if contrast_enhancer is not None else ClaheEnhancer(clip_limit=2.0)
        self.sharpener = sharpener if sharpener is not None else UnsharpMaskEnhancer(amount=1.0)

    def process(
        self,
        frame: np.ndarray,
        previous_frames: Optional[List[np.ndarray]] = None,
        timestamps: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.perf_counter()

        # Stage 1: Denoise
        f1, m1 = self.denoiser.process(frame, previous_frames, timestamps)

        # Stage 2: Illumination/Contrast
        f2, m2 = self.contrast_enhancer.process(f1, previous_frames, timestamps)

        # Stage 3: Sharpen
        enhanced_frame, m3 = self.sharpener.process(f2, previous_frames, timestamps)

        elapsed_time = time.perf_counter() - start_time

        metadata = {
            "processor": "CompositeEnhancer",
            "category": "Composite Classical",
            "stages": [
                m1.get("processor"),
                m2.get("processor"),
                m3.get("processor"),
            ],
            "processing_time_sec": elapsed_time,
            "shape": enhanced_frame.shape,
            "dtype": str(enhanced_frame.dtype),
        }

        return enhanced_frame, metadata
