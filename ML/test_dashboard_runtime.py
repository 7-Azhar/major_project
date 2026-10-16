"""Regressions for paused review, stale work, and bounded live capture."""
from pathlib import Path
import threading
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from dashboard import Pipeline, DEFAULT_SOURCE
from frame_processor import FrameProcessor
from video_sources import LatestFrameCapture


def test_review_controls_reuse_predictions_but_new_inputs_do_not():
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = Mock()
    boxes.xyxy.cpu.return_value.tolist.return_value = [[10, 220, 40, 300]]
    boxes.conf.cpu.return_value.tolist.return_value = [.4]
    boxes.cls.cpu.return_value.tolist.return_value = [0]
    result = Mock(boxes=boxes)
    # Mock special methods explicitly, so plotting has a real image.
    result.__getitem__ = Mock(return_value=Mock(plot=Mock(return_value=raw.copy())))
    model = Mock(names={0: 'pedestrian'})
    model.predict.return_value = [result]
    enhancer = Mock()
    enhancer.process.side_effect = lambda frame: (frame, {})
    factory = Mock(return_value=enhancer)
    hardware = Mock(cuda=False, fp16=False, predict_device='cpu')
    processor = FrameProcessor(cv2, model, hardware, factory,
                               Path(__file__).parent / 'profiles/user_flight_osd.json')
    cfg = dict(enhancement='identity', detection='original', outlines=False)
    first = processor.process(raw, cfg, new_frame=True)
    assert not first['inference_reused'] and len(first['detections']) == 1
    cfg['detection'] = 'exclude50'
    filtered = processor.process(raw, cfg, new_frame=False)
    assert filtered['inference_reused'] and filtered['detections'] == []
    assert filtered['excluded'][0]['excluded_by'] == 'confidence'
    assert filtered['original'] == first['original']
    cfg['detection'] = 'off'
    off = processor.process(raw, cfg, new_frame=False)
    assert off['detection_ms'] == 0 and not off['detections']
    cfg.update(detection='original', outlines=True)
    restored = processor.process(raw, cfg, new_frame=False)
    assert restored['detections'] == first['detections']
    assert model.predict.call_count == enhancer.process.call_count == 1
    processor.process(raw, cfg, new_frame=True)
    assert model.predict.call_count == enhancer.process.call_count == 2
    cfg['enhancement'] = 'clahe'
    processor.process(raw, cfg, new_frame=False)
    assert model.predict.call_count == enhancer.process.call_count == 3
    processor.reset()  # seek or a different recording, even with identical pixels
    processor.process(raw, cfg, new_frame=True)
    assert model.predict.call_count == 4 and factory.call_count == 3


def test_live_reader_discards_backlog_and_releases_once():
    class Capture:
        def __init__(self):
            self.step = 0
            self.blocked, self.finish = threading.Event(), threading.Event()
            self.release = Mock()

        def read(self):
            self.step += 1
            if self.step <= 20:
                return True, self.step
            self.blocked.set()
            self.finish.wait(2)
            return False, None

    capture = Capture()
    closed = Mock()
    reader = LatestFrameCapture(capture, closed, timeout=.01)
    try:
        assert capture.blocked.wait(1)
        assert reader.read() == (True, 20)
        assert reader.skipped == 19
        assert reader.read() == (False, None)
    finally:
        capture.finish.set()
        reader.release()
    reader.release()
    capture.release.assert_called_once()
    closed.assert_called_once()
    assert not reader.thread.is_alive()


def make_pipeline():
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.condition = threading.Condition()
    pipeline.stop = False
    pipeline.server_id, pipeline.sequence, pipeline.revision = 'server-a', 1, 4
    pipeline.config = dict(source=DEFAULT_SOURCE, playing=False)
    pipeline.snapshot = dict(sequence=1, server_id='server-a', revision=3,
                             total_frames=10, config=dict(source='previous-video'))
    pipeline.sources = [DEFAULT_SOURCE]
    return pipeline


def test_stale_work_cannot_overwrite_new_settings_and_idle_frames_are_unchanged():
    pipeline = make_pipeline()
    with pipeline.condition:
        assert not pipeline.publish(dict(status='ended'), 3)
        assert pipeline.sequence == 1
    assert pipeline.get_snapshot(1, 'server-a', timeout=.001) is None
    with pipeline.condition:
        assert pipeline.publish(dict(status='paused'), 4)
    assert pipeline.get_snapshot(1, 'server-a')['sequence'] == 2
    assert pipeline.get_snapshot(2, 'old-server')['server_id'] == 'server-a'


def test_seek_uses_current_source_bounds_not_previous_source():
    pipeline = make_pipeline()
    pipeline.update(dict(seek=500))  # old video's 10-frame length is irrelevant
    assert pipeline.seek == 500
    pipeline.snapshot['config']['source'] = DEFAULT_SOURCE
    with pytest.raises(ValueError, match='outside'):
        pipeline.update(dict(seek=500))


def test_invalid_enhancement_is_a_validation_error():
    pipeline = make_pipeline()
    with pytest.raises(ValueError, match='enhancement'):
        pipeline.update(dict(enhancement=[]))
