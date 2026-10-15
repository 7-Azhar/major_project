"""Compare original predictions and explicit post-filters without rerunning inference."""
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np
from osd_filter import OsdFilter


def draw(frame, detections, title, osd=None):
    result = frame.copy()
    if osd:
        osd.draw_regions(result)
    for item in detections:
        x1, y1, x2, y2 = map(int, item['xyxy'])
        color = (70, 230, 90) if item['class_id'] in [0, 1] else (240, 180, 60)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 1)
        label = f"{item['label']} {item['confidence']:.2f}"
        cv2.putText(result, label, (min(x1, 480), max(42, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
    cv2.rectangle(result, (0, 0), (640, 32), (0, 0, 0), -1)
    cv2.putText(result, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
    return result


def main():
    root = Path(__file__).resolve().parent
    source = root.parent / 'feeed enhancement/samples/fpv/real_fpv_user_flight.mp4'
    original = root / 'outputs/visdrone_raw_640_full'
    output = root / 'outputs/osd_review'
    output.mkdir(exist_ok=False)
    osd = OsdFilter(root / 'profiles/user_flight_osd.json', 640, 480)
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(str(output / 'comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1920,480))
    if not writer.isOpened():
        raise RuntimeError('Cannot write comparison')
    stats = Counter()
    reasons = Counter()
    snapshots = {0, 100, 250, 500, 1000, 1750, 1850, 2000, 3000, 4000}
    try:
        with (original / 'detections.jsonl').open() as records, (output / 'decisions.jsonl').open('w') as decisions:
            for line in records:
                record = json.loads(line)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError('Video ended before the prediction log')
                raw = record['detections']
                kept, excluded = osd.apply(raw)
                conservative = [d for d in kept if d['confidence'] >= .5]
                stats.update(frames=1, original_boxes=len(raw), osd_retained=len(kept),
                             osd_excluded=len(excluded), osd_conf50_retained=len(conservative),
                             original_human_boxes=sum(d['class_id'] in [0,1] for d in raw),
                             osd_human_boxes=sum(d['class_id'] in [0,1] for d in kept),
                             osd_conf50_human_boxes=sum(d['class_id'] in [0,1] for d in conservative))
                reasons.update(d['excluded_by'] for d in excluded)
                decisions.write(json.dumps(dict(frame=record['frame'], retained=kept, excluded=excluded,
                                                confidence50_retained=conservative)) + '\n')
                if record['frame'] < 750 or 1750 <= record['frame'] < 2050 or record['frame'] in snapshots:
                    panels = [draw(frame, raw, 'Original model | confidence >= 0.25'),
                              draw(frame, kept, 'OSD exclusion | confidence >= 0.25', osd),
                              draw(frame, conservative, 'OSD exclusion | confidence >= 0.50', osd)]
                    canvas = np.hstack(panels)
                    if record['frame'] < 750 or 1750 <= record['frame'] < 2050:
                        cv2.putText(canvas, f'Source frame {record["frame"]} | Amber outlines = configured exclusion regions',
                                    (8,470), cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                        writer.write(canvas)
                    if record['frame'] in snapshots:
                        cv2.imwrite(str(output / f'comparison_{record["frame"]:06d}.jpg'),canvas)
    finally:
        capture.release()
        writer.release()
    report = dict(counts=dict(stats), excluded_by_region=dict(reasons),
                  source_predictions=str(original / 'detections.jsonl'),
                  comparison_segments=[[0,749],[1750,2049]],
                  note='Same cached predictions in all panels. These counts measure filtering, not accuracy. No new inference benchmark. Small real targets in outlined regions may be excluded; confidence filtering may lose true detections. STATS layout after flight is not handled.')
    (output / 'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
