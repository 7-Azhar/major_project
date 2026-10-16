"""Unit tests for frame processing pipeline and Identity baseline processor."""

import numpy as np
import pytest
from app.pipeline import IdentityProcessor


def test_identity_processor_pass_through():
    """Verify that IdentityProcessor preserves dimensions, channels, dtype, and exact array content."""
    processor = IdentityProcessor()

    # Create test synthetic frame (720x1280x3 uint8)
    dummy_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    original_copy = dummy_frame.copy()

    output_frame, metadata = processor.process(dummy_frame)

    assert output_frame.shape == (720, 1280, 3)
    assert output_frame.dtype == np.uint8
    assert metadata["processor"] == "IdentityProcessor"
    assert "processing_time_sec" in metadata
    assert metadata["processing_time_sec"] >= 0.0

    # Ensure array content is preserved exactly
    assert np.array_equal(output_frame, original_copy)


def test_identity_processor_different_shapes():
    """Verify IdentityProcessor handles various image dimensions."""
    processor = IdentityProcessor()

    for shape in [(100, 100, 3), (480, 640, 3), (1080, 1920, 3)]:
        frame = np.zeros(shape, dtype=np.uint8)
        out, meta = processor.process(frame)
        assert out.shape == shape
        assert np.array_equal(out, frame)
