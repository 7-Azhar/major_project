"""Take pixels and metadata under one lock for recording and incident evidence."""
import time


def capture_evidence(video, *, copy_frame=False):
    with video.lock:
        state = video.snapshot()
        matched = state.get('backend') == 'aura' or state.get('detection_sequence') == video.sequence
        if not matched:
            state = dict(state, detections=[])
        frame = video.frame
        return dict(state=state, jpeg=video.jpeg,
                    frame=frame.copy() if copy_frame and frame is not None else frame,
                    sequence=video.sequence, detections_match_frame=matched,
                    captured_age_s=max(0, time.monotonic()-video.frame_at) if video.frame_at else None,
                    source_id=state.get('source_id', state.get('source', 'native')))
