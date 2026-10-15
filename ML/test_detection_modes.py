from detection_modes import select_detections


class Overlay:
    def reason(self, detection):
        return 'battery' if detection['label'] == 'overlay' else None


def test_switch_back_restores_original_candidates():
    raw = [dict(label='overlay', confidence=.9), dict(label='person', confidence=.49),
           dict(label='person', confidence=.5)]
    assert select_detections(raw, 0, Overlay())[0] == [0, 1, 2]
    assert select_detections(raw, 1, Overlay())[0] == [1, 2]
    assert select_detections(raw, 2, Overlay())[0] == [2]
    assert select_detections(raw, 0, Overlay())[0] == [0, 1, 2]
    assert all('excluded_by' not in item for item in raw)
