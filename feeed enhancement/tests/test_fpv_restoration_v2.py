"""Unit tests for Phase 5B FpvRestorationV2Enhancer."""

import numpy as np
import pytest

from app.enhancers.fpv_restoration_v2 import FpvRestorationV2Enhancer


def test_fpv_restoration_v2_output_shape_dtype_range():
    """Test output shape, dtype, and intensity range [0, 255]."""
    enhancer = FpvRestorationV2Enhancer()
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    enhanced, meta = enhancer.process(frame)
    assert enhanced is not None
    assert enhanced.shape == frame.shape
    assert enhanced.dtype == np.uint8
    assert np.min(enhanced) >= 0
    assert np.max(enhanced) <= 255
    assert meta["processor"] == "FpvRestorationV2Enhancer"
    assert "latency_ms" in meta
    assert meta["latency_ms"] >= 0.0


def test_fpv_restoration_v2_presets():
    """Test preset initialization (SAFE, BALANCED, DETAIL)."""
    for preset_name in ["SAFE", "BALANCED", "DETAIL"]:
        enhancer = FpvRestorationV2Enhancer(preset=preset_name)
        assert enhancer.preset_name == preset_name
        frame = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        enhanced, meta = enhancer.process(frame)
        assert enhanced.shape == (240, 320, 3)
        assert meta["preset"] == preset_name


def test_fpv_restoration_v2_deterministic_output():
    """Test that identical input frames produce identical outputs when state is reset."""
    enhancer = FpvRestorationV2Enhancer(preset="BALANCED")
    frame = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)

    enhancer.reset()
    out1, _ = enhancer.process(frame)

    enhancer.reset()
    out2, _ = enhancer.process(frame)

    assert np.array_equal(out1, out2)


def test_fpv_restoration_v2_parameter_bounds():
    """Test custom strength bounds and metadata output."""
    enhancer = FpvRestorationV2Enhancer(
        deblock_strength=0.8,
        denoise_strength=0.9,
        detail_strength=0.3,
        contrast_strength=0.2,
        temporal_stability=0.85,
    )
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)

    enhanced, meta = enhancer.process(frame)
    assert enhanced.shape == (100, 100, 3)
    assert meta["parameters"]["deblock_strength"] == 0.8
    assert meta["parameters"]["detail_strength"] == 0.3


def test_fpv_restoration_v2_invalid_input_raises():
    """Test that None or empty frame raises ValueError."""
    enhancer = FpvRestorationV2Enhancer()
    with pytest.raises(ValueError):
        enhancer.process(None)

    with pytest.raises(ValueError):
        enhancer.process(np.array([], dtype=np.uint8))
