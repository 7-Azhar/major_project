"""Unit tests for Phase 5A FpvHybridEnhancer."""

import numpy as np
import pytest

from app.enhancers.fpv_hybrid import FpvHybridEnhancer


def test_fpv_hybrid_output_shape_dtype_range():
    """Test output shape, dtype, and intensity range [0, 255]."""
    enhancer = FpvHybridEnhancer()
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    enhanced, meta = enhancer.process(frame)
    assert enhanced is not None
    assert enhanced.shape == frame.shape
    assert enhanced.dtype == np.uint8
    assert np.min(enhanced) >= 0
    assert np.max(enhanced) <= 255
    assert meta["processor"] == "FpvHybridEnhancer"
    assert "latency_ms" in meta
    assert meta["latency_ms"] >= 0.0


def test_fpv_hybrid_deterministic_output():
    """Test that identical input frames produce identical outputs."""
    enhancer = FpvHybridEnhancer(
        bilateral_strength=1.0,
        restoration_strength=1.0,
        contrast_strength=0.2,
        detail_strength=0.2,
    )
    frame = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)

    out1, _ = enhancer.process(frame)
    out2, _ = enhancer.process(frame)

    assert np.array_equal(out1, out2)


def test_fpv_hybrid_parameter_bounds():
    """Test custom parameter strength bounds."""
    enhancer = FpvHybridEnhancer(
        bilateral_strength=0.5,
        restoration_strength=0.8,
        contrast_strength=0.1,
        detail_strength=0.1,
    )
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)

    enhanced, meta = enhancer.process(frame)
    assert enhanced.shape == (100, 100, 3)
    assert meta["parameters"]["bilateral_strength"] == 0.5
    assert meta["parameters"]["contrast_strength"] == 0.1


def test_fpv_hybrid_invalid_input_raises():
    """Test that None or empty frame raises ValueError."""
    enhancer = FpvHybridEnhancer()
    with pytest.raises(ValueError):
        enhancer.process(None)

    with pytest.raises(ValueError):
        enhancer.process(np.array([], dtype=np.uint8))
