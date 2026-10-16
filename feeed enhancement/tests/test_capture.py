"""Unit tests for capture layer and error handling."""

import os
import pytest
from app.capture import FrameCapture
from scripts.generate_sample_video import generate_synthetic_video


def test_capture_invalid_file_raises():
    """Verify FrameCapture raises ValueError for missing video file."""
    capture = FrameCapture(source="non_existent_file_xyz_123.mp4")
    with pytest.raises(ValueError):
        capture.start()


def test_capture_synthetic_file(tmp_path):
    """Verify FrameCapture correctly reads frames and metadata from a synthetic video file."""
    test_video_path = str(tmp_path / "test_synth.mp4")
    generate_synthetic_video(output_path=test_video_path, width=640, height=360, fps=30, duration_sec=1)

    capture = FrameCapture(source=test_video_path, queue_size=5)
    capture.start()

    assert capture.is_opened
    assert capture.width == 640
    assert capture.height == 360
    assert capture.reported_fps > 0

    success, frame_id, ts, frame = capture.read(timeout=1.0)
    assert success
    assert frame_id == 1
    assert frame is not None
    assert frame.shape == (360, 640, 3)
    assert ts > 0

    capture.stop()
    assert not capture.is_opened


def test_capture_end_of_stream(tmp_path):
    """Verify FrameCapture detects End-of-Stream (EOS)."""
    test_video_path = str(tmp_path / "test_eos.mp4")
    # Generate 5 frames video (approx 0.16s at 30 fps)
    generate_synthetic_video(output_path=test_video_path, width=320, height=240, fps=30, duration_sec=0.2)

    capture = FrameCapture(source=test_video_path, queue_size=10)
    capture.start()

    read_count = 0
    while True:
        success, frame_id, ts, frame = capture.read(timeout=0.2)
        if not success:
            break
        read_count += 1

    assert read_count > 0
    assert capture.is_eos
    capture.stop()
