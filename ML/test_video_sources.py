from pathlib import Path
import threading
from unittest.mock import Mock
import pytest
from video_sources import LatestFrameCapture, is_camera, open_source
from dashboard import Pipeline, DEFAULT_SOURCE


def test_camera_ids_are_bounded():
    assert is_camera('camera:0') and is_camera('camera:9')
    assert not is_camera('camera:-1') and not is_camera('camera:10')
    assert not is_camera('../camera:0')


def test_failed_camera_releases_device():
    cv = Mock()
    cv.VideoCapture.return_value.isOpened.return_value = False
    with pytest.raises(RuntimeError, match='unavailable'):
        open_source(cv, Path('.'), 'camera:1')
    cv.VideoCapture.return_value.release.assert_called_once()


def test_camera_capture_requests_small_frames():
    cv = Mock()
    cv.VideoCapture.return_value.isOpened.return_value = True
    capture = open_source(cv, Path('.'), 'camera:2')
    assert cv.VideoCapture.call_args.args[0] == 2
    assert capture.set.call_count == 3


def test_live_camera_rejects_seek_and_preserves_selected_filter():
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.sources = [DEFAULT_SOURCE]
    pipeline.condition = threading.Condition()
    pipeline.config = dict(source=DEFAULT_SOURCE, detection='exclude25', playing=False)
    pipeline.snapshot = dict(total_frames=4496)
    pipeline.revision = 0
    pipeline.update(dict(source='camera:0', playing=True))
    assert pipeline.seek is None
    assert pipeline.config['detection'] == 'exclude25'
    with pytest.raises(ValueError, match='cannot seek'):
        pipeline.update(dict(seek=0))


def test_buffered_camera_frame_keeps_capture_time(monkeypatch):
    capture = Mock()
    capture.read.side_effect = [(True, 'frame'), (False, None)]
    monkeypatch.setattr('video_sources.time.time', lambda: 123)
    buffered = LatestFrameCapture(capture)
    buffered.thread.join(2)
    monkeypatch.setattr('video_sources.time.time', lambda: 999)
    assert buffered.read() == (True, 'frame')
    assert buffered.delivered_at == 123
    buffered.release()
