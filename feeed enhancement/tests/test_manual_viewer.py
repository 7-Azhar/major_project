"""Unit tests for Phase 4A Manual Real-Time Video Enhancement Viewer."""

import json
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from app.manual_viewer import ManualViewerApp, ZeroDceOnnxCUDAEnhancer
from app.pipeline import IdentityProcessor

SAMPLE_VIDEO = str(Path(__file__).resolve().parents[1] / 'samples' / 'test_synthetic.mp4')


def test_zero_dce_onnx_cuda_enhancer_fallback():
    """Test Zero-DCE ONNX enhancer instantiates and processes a synthetic frame."""
    enhancer = ZeroDceOnnxCUDAEnhancer()
    frame = np.random.randint(0, 256, (360, 640, 3), dtype=np.uint8)

    enhanced, meta = enhancer.process(frame)
    assert enhanced is not None
    assert enhanced.shape == frame.shape
    assert enhanced.dtype == np.uint8
    assert "processor" in meta
    assert "latency_ms" in meta


def test_manual_viewer_display_modes():
    """Test frame composition across all viewer modes."""
    app = ManualViewerApp(
        source=SAMPLE_VIDEO,
        headless=True,
        max_frames=5,
    )

    raw_frame = np.full((360, 640, 3), 100, dtype=np.uint8)
    enhanced_frame = np.full((360, 640, 3), 200, dtype=np.uint8)

    # 1. Original
    app.current_mode = "original"
    comp_orig = app._compose_display_frame(raw_frame, enhanced_frame)
    assert comp_orig.shape == (360, 640, 3)
    assert np.all(comp_orig == 100)

    # 2. Enhanced
    app.current_mode = "enhanced"
    comp_enh = app._compose_display_frame(raw_frame, enhanced_frame)
    assert comp_enh.shape == (360, 640, 3)
    assert np.all(comp_enh == 200)

    # 3. Side-by-Side
    app.current_mode = "side_by_side"
    comp_sbs = app._compose_display_frame(raw_frame, enhanced_frame)
    assert comp_sbs.shape == (360, 1280, 3)  # Width doubled

    # 4. Split View
    app.current_mode = "split_view"
    app.split_ratio = 0.5
    comp_split = app._compose_display_frame(raw_frame, enhanced_frame)
    assert comp_split.shape == (360, 640, 3)
    # Left half should be raw (100), right half should be enhanced (200) except divider line
    assert comp_split[180, 50, 0] == 100
    assert comp_split[180, 500, 0] == 200


def test_manual_viewer_enhancer_switching():
    """Test switching enhancers via keys 1-6."""
    app = ManualViewerApp(
        source=SAMPLE_VIDEO,
        headless=True,
        max_frames=5,
    )

    for key in ["1", "2", "3", "4", "5", "6"]:
        app.current_enhancer_key = key
        label, enhancer = app.get_current_enhancer()
        assert label is not None
        assert enhancer is not None


def test_manual_viewer_headless_run_and_logging():
    """Test full headless run on synthetic video and JSON summary generation."""
    temp_dir = tempfile.mkdtemp()
    try:
        app = ManualViewerApp(
            source=SAMPLE_VIDEO,
            initial_enhancer_key="6",
            initial_mode="side_by_side",
            output_dir=temp_dir,
            headless=True,
            max_frames=10,
        )

        app.run()

        # Check JSON output report created
        json_files = [f for f in os.listdir(temp_dir) if f.endswith(".json")]
        assert len(json_files) == 1

        json_path = os.path.join(temp_dir, json_files[0])
        with open(json_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        assert "timestamp" in report
        assert "source" in report
        assert "average_fps" in report
        assert "median_processing_latency_ms" in report
        assert report["frame_count"] == 10
        assert len(report["evaluated_enhancers"]) > 0

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
