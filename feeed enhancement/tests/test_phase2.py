"""Unit tests for Phase 2 neural enhancer and downstream AI evaluation."""

import numpy as np
import pytest

from app.enhancers import get_enhancer_by_name
from app.enhancers.neural import ZeroDceEnhancer
from app.eval import BoundingBox, YoloEvaluator


@pytest.fixture
def sample_frame():
    """Generates synthetic 480x640x3 uint8 image frame."""
    np.random.seed(42)
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


def test_zerodce_enhancer(sample_frame):
    """Verify ZeroDceEnhancer preserves dimensions, dtype, and returns valid metadata."""
    enhancer = ZeroDceEnhancer(device="cpu", fp16=False, num_curves=4)
    enhanced, meta = enhancer.process(sample_frame)

    assert enhanced.shape == sample_frame.shape
    assert enhanced.dtype == np.uint8
    assert meta["processor"] == "ZeroDceEnhancer"
    assert meta["category"] == "Lightweight Neural Illumination"
    assert meta["processing_time_sec"] >= 0.0


def test_zerodce_factory_lookup():
    """Verify factory lookup for neural & zerodce names."""
    proc1 = get_enhancer_by_name("neural")
    proc2 = get_enhancer_by_name("zerodce")

    assert proc1.__class__.__name__ == "ZeroDceEnhancer"
    assert proc2.__class__.__name__ == "ZeroDceEnhancer"


def test_yolo_evaluator_iou():
    """Verify IoU metric calculation in YoloEvaluator."""
    b1 = BoundingBox(0, "target", 0.9, 10.0, 10.0, 50.0, 50.0)
    b2 = BoundingBox(0, "target", 0.9, 10.0, 10.0, 50.0, 50.0)
    b3 = BoundingBox(0, "target", 0.9, 100.0, 100.0, 150.0, 150.0)

    iou_self = YoloEvaluator.compute_iou(b1, b2)
    iou_no_overlap = YoloEvaluator.compute_iou(b1, b3)

    assert pytest.approx(iou_self, 1e-4) == 1.0
    assert iou_no_overlap == 0.0


def test_yolo_evaluator_stream(sample_frame):
    """Verify downstream evaluation stream execution."""
    evaluator = YoloEvaluator()
    raw_list = [sample_frame, sample_frame]
    enh_list = [sample_frame, sample_frame]

    result = evaluator.evaluate_stream(raw_list, enh_list, processor_name="ZeroDceEnhancer")
    res_dict = result.to_dict()

    assert res_dict["evaluator_name"] == "YoloEvaluator"
    assert res_dict["processor_name"] == "ZeroDceEnhancer"
    assert res_dict["total_frames_evaluated"] == 2
    assert "precision" in res_dict
    assert "recall" in res_dict
    assert "map_50" in res_dict
