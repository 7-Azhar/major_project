"""Matched FPV screening of baseline and official COCO-pretrained models; no accuracy claims."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/pretrained_screening_01')
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from osd_filter import OsdFilter
    from detection_modes import select_detections
    if args.output.exists():
        raise RuntimeError('Choose a new output directory; results are never overwritten')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    args.output.mkdir(parents=True)
    source = ROOT.parent / 'feeed enhancement/samples/fpv/real_fpv_user_flight.mp4'
    indices = sorted(set(range(0, 750, 5)) | set(range(1750, 2050, 5)) | {543})
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    wanted = set(indices)
    for index in range(max(indices) + 1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Cannot decode frame {index}')
        if index in wanted:
            frames.append(frame)
    cap.release()
    height, width = frames[0].shape[:2]
    osd = OsdFilter(ROOT / 'profiles/user_flight_osd.json', width, height)
    candidates = [('experiment_1', 'yolov8m_visdrone_recovered.pt'), ('yolo11s', 'yolo11s.pt'),
                  ('yolo26n', 'yolo26n.pt'), ('yolo26s', 'yolo26s.pt')]
    reports, all_predictions = [], []
    for label, filename in candidates:
        print(f'Loading {label}', flush=True)
        path = ROOT / 'models' / filename
        # Official model filenames trigger Ultralytics' official asset downloader only here.
        model = YOLO(str(path)).to(0)
        options = dict(imgsz=640, conf=.25, iou=.7, device=0, quantize=16, verbose=False, max_det=300)
        for _ in range(5):
            model.predict(frames[0], **options)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times, predictions = [], []
        raw_counts, kept_counts = Counter(), Counter()
        with (args.output / f'{label}.jsonl').open('w', encoding='utf-8') as stream:
            for index, frame in zip(indices, frames):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = model.predict(frame, **options)[0]
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
                raw = [dict(class_id=int(c), label=model.names[int(c)], confidence=float(s), xyxy=b)
                       for b, s, c in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())]
                keep, excluded = select_detections(raw, 1, osd)
                retained = [raw[i] for i in keep]
                raw_counts.update(d['label'] for d in raw)
                kept_counts.update(d['label'] for d in retained)
                record = dict(frame=index, raw_detections=raw, detections=retained, excluded=excluded, predict_ms=times[-1])
                stream.write(json.dumps(record) + '\n')
                predictions.append(record)
        report = dict(model=label, weights=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      classes=model.names, frames=len(frames), median_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)),
                      peak_torch_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
                      raw_counts=dict(raw_counts), osd_counts=dict(kept_counts))
        reports.append(report)
        all_predictions.append(predictions)
        print(json.dumps(report), flush=True)
        del result, model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    def panel(frame, record, label):
        picture = frame.copy()
        osd.draw_regions(picture)
        for item in record['detections']:
            x1, y1, x2, y2 = map(int, item['xyxy'])
            human = item['label'] in ('person', 'people', 'pedestrian')
            color = (50, 255, 60) if human else (255, 160, 30)
            cv2.rectangle(picture, (x1, y1), (x2, y2), color, 1)
            cv2.putText(picture, f"{item['label']} {item['confidence']:.2f}", (max(0, x1), max(35, y1 - 3)), 0, .38, color, 1)
        cv2.rectangle(picture, (0, 0), (width, 24), (0, 0, 0), -1)
        cv2.putText(picture, f"{label} | Exclude .25 | frame {record['frame']}", (5, 17), 0, .45, (255, 255, 255), 1)
        return picture
    writer = cv2.VideoWriter(str(args.output / 'comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps / 5, (width * 2, height * 2))
    if not writer.isOpened():
        raise RuntimeError('Cannot open comparison writer')
    for n, (index, frame) in enumerate(zip(indices, frames)):
        panels = [panel(frame, all_predictions[m][n], candidates[m][0]) for m in range(4)]
        picture = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
        writer.write(picture)
        if index in (0, 500, 543, 1750, 1850, 2000):
            cv2.imwrite(str(args.output / f'comparison_{index:06d}.jpg'), picture)
    writer.release()
    report = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  frame_indices=indices, imgsz=640, conf=.25, iou=.7, quantize=16, warmup=5,
                  device=torch.cuda.get_device_name(0), torch=torch.__version__, ultralytics=ultralytics.__version__,
                  models=reports, limitations=[
                      'Screening on 211 selected frames; no ground-truth annotations or accuracy ranking.',
                      'COCO has 80 classes; VisDrone has 10. Counts are not directly comparable accuracy metrics.',
                      'Timing uses CUDA synchronization around predict, excludes decoding/drawing; sequential runs may vary thermally.',
                      'Video is a sampled review at source FPS / 5, with a jump from frame 745 to 1750 and an extra frame 543.',
                      'All panels apply the same optional OSD filter at .25; raw predictions remain in logs.',
                      'No retraining or enhancement. Torch memory excludes driver and other processes.'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
