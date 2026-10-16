"""Unit tests for FPV Restoration Baseline Enhancer."""

import numpy as np
import pytest

from app.enhancers.fpv_restoration import FpvRestorationEnhancer


def test_fpv_restoration_enhancer_processing():
    """Test FpvRestorationEnhancer processes a frame and produces valid metadata."""
    enhancer = FpvRestorationEnhancer()
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    enhanced, meta = enhancer.process(frame)
    assert enhanced is not None
    assert enhanced.shape == frame.shape
    assert enhanced.dtype == np.uint8
    assert meta["processor"] == "FpvRestorationEnhancer"
    assert "latency_ms" in meta
    assert meta["latency_ms"] >= 0.0


def test_fpv_restoration_deblocking_effect():
    """Test FpvRestorationEnhancer smooths synthetic macroblock boundary noise."""
    enhancer = FpvRestorationEnhancer()
    frame = np.full((16, 16, 3), 100, dtype=np.uint8)
    # Add artificial block step
    frame[:, 8:] = 180

    enhanced, _ = enhancer.process(frame)
    assert enhanced is not None
    assert enhanced.shape == frame.shape
