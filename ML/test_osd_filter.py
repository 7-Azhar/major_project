from pathlib import Path
import pytest
from osd_filter import OsdFilter


def profile():
    return OsdFilter(Path(__file__).parent / 'profiles/user_flight_osd.json', 640, 480)


def test_telemetry_excluded_but_person_extending_beyond_region_kept():
    detections = [dict(xyxy=[255, 355, 271, 389], confidence=.62, label='van'),
                  dict(xyxy=[250, 320, 290, 430], confidence=.51, label='pedestrian'),
                  dict(xyxy=[360, 415, 390, 470], confidence=.51, label='pedestrian')]
    kept, excluded = profile().apply(detections)
    assert kept == detections[1:]
    assert excluded[0]['excluded_by'] == 'battery_left'
    assert 'excluded_by' not in detections[0]


def test_small_target_in_overlay_zone_is_explicitly_excluded():
    # This documents the real blind spot, rather than pretending perfect recall.
    assert profile().reason(dict(xyxy=[220, 230, 230, 245])) == 'moving_horizon_marks'
    assert profile().reason(dict(xyxy=[220, 230, 250, 280])) is None


def test_wrong_resolution_rejected():
    with pytest.raises(ValueError, match='requires'):
        OsdFilter(Path(__file__).parent / 'profiles/user_flight_osd.json', 1280, 720)


@pytest.mark.parametrize('width,height', [(320, 240), (1280, 720), (1920, 1080)])
def test_dashboard_scales_regions_and_box_limits(width, height):
    scaled = OsdFilter(Path(__file__).parent / 'profiles/user_flight_osd.json', width, height, scale_to_frame=True)
    box = [255, 355, 271, 389]
    box = [v * (width / 640 if i % 2 == 0 else height / 480) for i, v in enumerate(box)]
    assert scaled.reason(dict(xyxy=box)) == 'battery_left'
    assert profile().profile['reference_size'] == [640, 480]
