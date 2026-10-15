"""Shared display-only modes: inference always retains scores >= 0.25."""
MODES = ('Original 0.25', 'Exclude 0.25', 'Exclude 0.50')


def select_detections(detections, mode, osd_filter):
    if mode not in range(3):
        raise ValueError('Unknown display mode')
    if mode and osd_filter is None:
        raise ValueError('Exclusion modes require a calibrated OSD profile')
    keep, excluded = [], []
    for index, detection in enumerate(detections):
        reason = osd_filter.reason(detection) if mode else None
        if reason is None and detection['confidence'] < (0.50 if mode == 2 else 0.25):
            reason = 'confidence'
        if reason:
            excluded.append(dict(detection, excluded_by=reason))
        else:
            keep.append(index)
    return keep, excluded


def render(frame, result, detections, mode, osd_filter, title, paused=False):
    import cv2
    import numpy as np
    keep, _ = select_detections(detections, mode, osd_filter)
    picture = result[keep].plot(font_size=12, line_width=1) if keep else frame.copy()
    if mode:
        osd_filter.draw_regions(picture)
    width = frame.shape[1]
    bar = np.zeros((94, width, 3), dtype=np.uint8)
    for i, label in enumerate(MODES):
        left, right = i * width // 3, (i + 1) * width // 3
        cv2.rectangle(bar, (left + 2, 2), (right - 2, 32),
                      (65, 110, 55) if i == mode else (55, 55, 55), -1)
        cv2.putText(bar, f'{i + 1}: {label}', (left + 7, 23), 0, .44, (255, 255, 255), 1)
    cv2.putText(bar, title, (6, 51), 0, .40, (255, 255, 255), 1)
    cv2.putText(bar, 'Space: pause/play | R: restart | Q: quit' + (' | PAUSED' if paused else ''),
                (6, 69), 0, .40, (255, 255, 255), 1)
    cv2.putText(bar, 'Amber regions may hide small real targets' if mode else 'Original predictions; exclusion disabled',
                (6, 87), 0, .39, (0, 190, 255), 1)
    return np.vstack((bar, picture))
