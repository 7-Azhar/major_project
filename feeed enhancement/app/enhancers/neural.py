"""Lightweight Zero-DCE (Zero-Reference Deep Curve Estimation) Neural Enhancer - Baseline 5.

Estimates 8-stage pixel-wise dynamic curve maps for low-light FPV video enhancement
without altering high-frequency spatial structures.
"""

import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.enhancers.neural_base import BaseNeuralEnhancer, HAS_TORCH

if HAS_TORCH:
    import torch
    import torch.nn as nn

    class ZeroDCENet(nn.Module):
        """Lightweight 7-layer Convolutional Curve Estimation Network."""

        def __init__(self, num_curves: int = 8):
            super().__init__()
            self.num_curves = num_curves
            self.relu = nn.ReLU(inplace=True)

            self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
            self.conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
            self.conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
            self.conv5 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
            self.conv6 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
            self.conv7 = nn.Conv2d(64, 3 * num_curves, 3, 1, 1, bias=True)
            self.tanh = nn.Tanh()

        def forward(self, x):
            x1 = self.relu(self.conv1(x))
            x2 = self.relu(self.conv2(x1))
            x3 = self.relu(self.conv3(x2))
            x4 = self.relu(self.conv4(x3))

            x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
            x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
            a = self.tanh(self.conv7(torch.cat([x1, x6], dim=1)))

            # Apply iterative curve transformation: LE_n = LE_{n-1} + a_n * LE_{n-1} * (1 - LE_{n-1})
            curr = x
            for i in range(self.num_curves):
                a_i = a[:, i * 3 : (i + 1) * 3, :, :]
                curr = curr + a_i * (torch.pow(curr, 2) - curr)
            return curr


class ZeroDceEnhancer(BaseNeuralEnhancer):
    """
    Zero-DCE (Baseline 5: Lightweight Neural Enhancement).

    Applies deep curve estimation for low-light enhancement.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        fp16: bool = True,
        num_curves: int = 8,
    ):
        super().__init__(model_path=model_path, device=device, fp16=fp16)
        self.num_curves = num_curves

        if not 1 <= num_curves <= 8:
            raise ValueError("Zero-DCE supports between 1 and 8 pretrained curve iterations")
        if model_path is None:
            model_path = str(Path(__file__).resolve().parents[2] / "outputs/models/Epoch99.pth")

        self.model_path = model_path

        if self.has_torch:
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(f"Pretrained Zero-DCE checkpoint is required: {self.model_path}")
            self.net = ZeroDCENet(num_curves=self.num_curves).to(self.device)
            if self.model_path and os.path.exists(self.model_path):
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                mapped_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace("e_conv", "conv")
                    if new_key in ("conv7.weight", "conv7.bias") and v.shape[0] == 24:
                        v = v[:3 * self.num_curves]
                    mapped_state_dict[new_key] = v
                self.net.load_state_dict(mapped_state_dict, strict=True)
            if self.fp16 and self.device_str == "cuda":
                self.net = self.net.half()
            self.net.eval()
        else:
            self.net = None

    def _forward_neural(self, tensor_in):
        """Executes forward curve estimation pass."""
        if self.net is not None:
            return self.net(tensor_in)
        return tensor_in

    def process(
        self,
        frame: np.ndarray,
        previous_frames: Optional[List[np.ndarray]] = None,
        timestamps: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.perf_counter()

        if self.has_torch and self.net is not None:
            tensor_in = self.preprocess(frame)
            with torch.no_grad():
                tensor_out = self._forward_neural(tensor_in)
            enhanced_frame = self.postprocess(tensor_out)
        else:
            # Analytical gamma curve estimation fallback if PyTorch is unavailable
            norm = frame.astype(np.float32) / 255.0
            gamma = 0.6
            enhanced_frame = (np.power(norm, gamma) * 255.0).clip(0, 255).astype(np.uint8)

        elapsed_time = time.perf_counter() - start_time

        metadata = {
            "processor": "ZeroDceEnhancer",
            "category": "Lightweight Neural Illumination",
            "device": self.device_str,
            "fp16": self.fp16,
            "num_curves": self.num_curves,
            "pretrained": self.net is not None,
            "method": "pretrained_zero_dce" if self.net is not None else "analytical_gamma",
            "processing_time_sec": elapsed_time,
            "shape": enhanced_frame.shape,
            "dtype": str(enhanced_frame.dtype),
        }

        return enhanced_frame, metadata
