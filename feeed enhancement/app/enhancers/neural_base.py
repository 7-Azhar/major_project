"""Abstract Neural Model Interface for AURA Video Engine.

Provides standard pre-processing, device placement (CPU/CUDA), FP16 half precision,
and post-processing pipelines for deep learning enhancement models.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.pipeline import BaseProcessor

logger = logging.getLogger(__name__)

# Safe PyTorch import check
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class BaseNeuralEnhancer(BaseProcessor):
    """
    Abstract Neural Network Enhancer Base Class.

    Inherit from this class when integrating deep learning enhancement models
    (e.g., Low-Light Enhancement, Denoising, Super-Resolution).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        fp16: bool = True,
    ):
        super().__init__()
        self.model_path = model_path
        self.fp16 = fp16
        self.has_torch = HAS_TORCH

        if not HAS_TORCH:
            self.device_str = "cpu"
            logger.warning("PyTorch is not installed. Neural enhancer running in mock/fallback mode.")
        else:
            if device == "auto":
                self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device_str = device

            self.device = torch.device(self.device_str)
            logger.info(f"BaseNeuralEnhancer initialized on device: '{self.device_str}' (FP16={self.fp16})")

    def preprocess(self, frame: np.ndarray):
        """Convert BGR uint8 numpy array to normalized float32/fp16 torch Tensor (1, C, H, W)."""
        if not HAS_TORCH:
            return frame

        # BGR -> RGB
        rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
        # (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.fp16 and self.device_str == "cuda":
            tensor = tensor.half()
        return tensor

    def postprocess(self, tensor_out) -> np.ndarray:
        """Convert float32/fp16 torch Tensor (1, C, H, W) back to BGR uint8 numpy array."""
        if not HAS_TORCH or not isinstance(tensor_out, torch.Tensor):
            return tensor_out

        tensor_out = tensor_out.squeeze(0).clamp(0.0, 1.0).float().cpu().numpy()
        # (C, H, W) -> (H, W, C)
        rgb = (tensor_out.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        # RGB -> BGR
        bgr = rgb[:, :, ::-1]
        return bgr

    def _forward_neural(self, tensor_in):
        """Override this method in subclasses to invoke model forward pass."""
        raise NotImplementedError("Subclasses must implement _forward_neural().")

    def process(
        self,
        frame: np.ndarray,
        previous_frames: Optional[List[np.ndarray]] = None,
        timestamps: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.perf_counter()

        if not HAS_TORCH:
            # Fallback identity if torch is absent
            output_frame = frame
            proc_name = f"{self.__class__.__name__} (Fallback Identity)"
        else:
            tensor_in = self.preprocess(frame)
            with torch.no_grad():
                tensor_out = self._forward_neural(tensor_in)
            output_frame = self.postprocess(tensor_out)
            proc_name = self.__class__.__name__

        elapsed_time = time.perf_counter() - start_time

        metadata = {
            "processor": proc_name,
            "category": "Neural Enhancement Interface",
            "device": self.device_str,
            "fp16": self.fp16,
            "processing_time_sec": elapsed_time,
            "shape": output_frame.shape,
            "dtype": str(output_frame.dtype),
        }

        return output_frame, metadata
