"""Optional, explicit FPV overlay exclusion; this is not a learned detector improvement."""
import json
from pathlib import Path


class OsdFilter:
    def __init__(self, profile, width, height, *, scale_to_frame=False):
        self.profile = json.loads(Path(profile).read_text(encoding='utf-8'))
        reference = self.profile['reference_size']
        if width <= 0 or height <= 0:
            raise ValueError('Frame dimensions must be positive')
        if [width, height] != reference and not scale_to_frame:
            raise ValueError(f'OSD profile requires {reference}, got {[width, height]}. Calibrate a new profile.')
        if scale_to_frame:
            sx, sy = width / reference[0], height / reference[1]
            for region in self.profile['regions']:
                region['xyxy'] = [v * (sx if i % 2 == 0 else sy) for i, v in enumerate(region['xyxy'])]
                region['max_box_size'] = [region['max_box_size'][0] * sx, region['max_box_size'][1] * sy]
            self.profile['reference_size'] = [width, height]
        self.regions = self.profile['regions']
        for region in self.regions:
            x1, y1, x2, y2 = region['xyxy']
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f'Invalid OSD region: {region}')

    def reason(self, detection):
        x1, y1, x2, y2 = detection['xyxy']
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            return 'invalid_box'
        for region in self.regions:
            max_width, max_height = region['max_box_size']
            if width > max_width or height > max_height:
                continue
            a, b, c, d = region['xyxy']
            overlap = max(0, min(x2, c) - max(x1, a)) * max(0, min(y2, d) - max(y1, b))
            if overlap / (width * height) >= region.get('min_box_overlap', 0.8):
                return region['name']
        return None

    def apply(self, detections):
        kept, excluded = [], []
        for detection in detections:
            reason = self.reason(detection)
            if reason:
                excluded.append(dict(detection, excluded_by=reason))
            else:
                kept.append(detection)
        return kept, excluded

    def draw_regions(self, image):
        import cv2
        for region in self.regions:
            x1, y1, x2, y2 = map(round, region['xyxy'])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 255), 1)
        return image
