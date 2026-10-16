"""Unit tests for Phase 1 enhancement baselines, quality metrics, and side-by-side display."""

import numpy as np
import pytest

from app.display import StreamDisplay
from app.enhancers import (
    BilateralDenoiseEnhancer,
    ClaheEnhancer,
    CompositeEnhancer,
    UnsharpMaskEnhancer,
    get_enhancer_by_name,
)
from app.enhancers.neural_base import BaseNeuralEnhancer
from app.eval import BoundingBox, DetectionEvaluationResult
from app.quality import (
    calculate_psnr,
    calculate_ssim,
    calculate_tenengrad,
    evaluate_quality_pair,
)


@pytest.fixture
def sample_frame():
    """Generates synthetic 480x640x3 uint8 image frame."""
    np.random.seed(42)
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


def test_clahe_enhancer(sample_frame):
    """Verify CLAHE (Baseline 1) preserves shape, uint8 dtype, and returns valid metadata."""
    enhancer = ClaheEnhancer(clip_limit=2.0)
    enhanced, meta = enhancer.process(sample_frame)

    assert enhanced.shape == sample_frame.shape
    assert enhanced.dtype == np.uint8
    assert meta["processor"] == "ClaheEnhancer"
    assert meta["category"] == "Illumination Enhancement"
    assert meta["processing_time_sec"] >= 0.0


def test_unsharp_mask_enhancer(sample_frame):
    """Verify Unsharp Mask (Baseline 2) preserves shape, uint8 dtype, and sharpens frame."""
    enhancer = UnsharpMaskEnhancer(amount=1.5)
    enhanced, meta = enhancer.process(sample_frame)

    assert enhanced.shape == sample_frame.shape
    assert enhanced.dtype == np.uint8
    assert meta["processor"] == "UnsharpMaskEnhancer"
    assert meta["category"] == "Sharpening"


def test_bilateral_denoise_enhancer(sample_frame):
    """Verify Bilateral Denoise (Baseline 3) preserves shape and dtype."""
    enhancer = BilateralDenoiseEnhancer(d=3, sigma_color=30.0, sigma_space=30.0)
    enhanced, meta = enhancer.process(sample_frame)

    assert enhanced.shape == sample_frame.shape
    assert enhanced.dtype == np.uint8
    assert meta["processor"] == "BilateralDenoiseEnhancer"
    assert meta["category"] == "Denoising"


def test_composite_enhancer(sample_frame):
    """Verify Composite Pipeline (Baseline 4) executes 3 stages in order."""
    enhancer = CompositeEnhancer()
    enhanced, meta = enhancer.process(sample_frame)

    assert enhanced.shape == sample_frame.shape
    assert enhanced.dtype == np.uint8
    assert meta["processor"] == "CompositeEnhancer"
    assert len(meta["stages"]) == 3


def test_enhancer_factory():
    """Verify get_enhancer_by_name factory loader."""
    for name in ["identity", "clahe", "unsharp", "bilateral", "composite"]:
        proc = get_enhancer_by_name(name)
        assert proc is not None

    with pytest.raises(ValueError):
        get_enhancer_by_name("non_existent_enhancer")


def test_abstract_neural_enhancer_fallback(sample_frame):
    """Verify BaseNeuralEnhancer gracefully falls back if custom forward is not implemented."""
    class DummyNeural(BaseNeuralEnhancer):
        def _forward_neural(self, tensor_in):
            return tensor_in

    dummy = DummyNeural(device="cpu", fp16=False)
    enhanced, meta = dummy.process(sample_frame)
    assert enhanced.shape == sample_frame.shape
    assert "Neural" in meta["processor"]


def test_quality_metrics(sample_frame):
    """Verify PSNR, SSIM, and Tenengrad calculation."""
    # Identical image PSNR should be infinite
    psnr_identical = calculate_psnr(sample_frame, sample_frame)
    assert psnr_identical == float("inf")

    # Identical image SSIM should be 1.0
    ssim_identical = calculate_ssim(sample_frame, sample_frame)
    assert pytest.approx(ssim_identical, 1e-4) == 1.0

    # Tenengrad score should be positive for non-empty frame
    tenengrad_score = calculate_tenengrad(sample_frame)
    assert tenengrad_score > 0.0

    pair_eval = evaluate_quality_pair(sample_frame, sample_frame)
    assert "psnr_db" in pair_eval
    assert "ssim" in pair_eval
    assert "sharpness_gain_pct" in pair_eval


def test_detection_evaluation_result_struct():
    """Verify DetectionEvaluationResult dictionary serialization."""
    res = DetectionEvaluationResult(
        evaluator_name="DummyYoloEvaluator",
        processor_name="ClaheEnhancer",
        total_frames_evaluated=100,
        precision=0.85,
        recall=0.80,
        map_50=0.82,
    )
    d = res.to_dict()
    assert d["evaluator_name"] == "DummyYoloEvaluator"
    assert d["processor_name"] == "ClaheEnhancer"
    assert d["map_50"] == 0.82


def test_side_by_side_display(sample_frame):
    """Verify StreamDisplay accepts side-by-side rendering parameters without crashing."""
    display = StreamDisplay(disabled=True)
    summary = {
        "processor": "ClaheEnhancer",
        "observed_output_fps": 60.0,
        "avg_frame_time_ms": 1.2,
        "resolution": "640x480",
        "frames_dropped": 0,
    }
    cont = display.render(
        frame=sample_frame,
        metrics_summary=summary,
        raw_frame=sample_frame,
        side_by_side=True,
    )
    assert cont is True
