"""Standalone AURA VisDrone video inference with synchronized timing and explicit device checks."""
import argparse
import hashlib
import json
from pathlib import Path
import time

VISDRONE_CLASSES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
                    'tricycle', 'awning-tricycle', 'bus', 'motor']


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=root / 'models/yolov8m_visdrone_recovered.pt')
    parser.add_argument('--source', type=Path, default=root.parent / 'feeed enhancement/samples/fpv/real_fpv_user_flight.mp4')
    parser.add_argument('--output', type=Path, default=root / 'outputs/visdrone_raw_640')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.7)
    parser.add_argument('--max-frames', type=int, default=750, help='0 processes the whole recording')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--osd-profile', type=Path, help='Optional calibrated OSD exclusion JSON; amber outlines show affected regions')
    args = parser.parse_args()
    if args.stride < 1 or args.max_frames < 0 or args.start_frame < 0:
        parser.error('stride must be positive; max-frames and start-frame must be nonnegative')
    if not args.weights.is_file() or not args.source.is_file():
        parser.error('Existing weights and recording are required; automatic downloads are disabled')
    if (args.output / 'report.json').exists() or (args.output / 'detections.jsonl').exists():
        parser.error('Output already contains results; choose a new --output directory')

    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. No silent CPU fallback. Use --device cpu explicitly.')
    device = 0 if args.device == 'cuda' else 'cpu'
    model = YOLO(str(args.weights), task='detect')
    names = [model.names[i] for i in range(len(model.names))]
    if names != VISDRONE_CLASSES:
        raise ValueError(f'Unexpected class mapping: {names}')
    attention = [(name, type(module).__name__) for name, module in model.model.named_modules()
                 if any(term in type(module).__name__.lower() for term in ['cbam', 'attention'])]
    model.to(device)
    actual_device = str(next(model.model.parameters()).device)
    if args.device == 'cuda' and not actual_device.startswith('cuda'):
        raise RuntimeError(f'Model device mismatch: {actual_device}')
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {args.source}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    if not ok or fps <= 0:
        cap.release()
        raise RuntimeError('Recording has no readable first frame or valid frame rate')
    height, width = first.shape[:2]
    from osd_filter import OsdFilter
    osd_filter = OsdFilter(args.osd_profile, width, height) if args.osd_profile else None
    options = dict(imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=device,
                   quantize=16 if args.device == 'cuda' else 32, verbose=False, max_det=300)
    for _ in range(5):
        model.predict(first, **options)
    def synchronize():
        if args.device == 'cuda':
            torch.cuda.synchronize()
    synchronize()
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    if args.start_frame >= total_frames:
        cap.release()
        raise ValueError('start-frame is beyond the recording')
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame):
        cap.release()
        raise RuntimeError('Cannot seek to requested frame')
    args.output.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(str(args.output / 'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps / args.stride, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError('Cannot create annotated video')
    timings, stage_times, counts = [], [], {name: 0 for name in names}
    excluded_counts = {}
    index, processed = args.start_frame - 1, 0
    started = time.perf_counter()
    try:
        with (args.output / 'detections.jsonl').open('w', encoding='utf-8') as stream:
            while not args.max_frames or processed < args.max_frames:
                ok, frame = cap.read()
                index += 1
                if not ok:
                    break
                if (index - args.start_frame) % args.stride:
                    continue
                synchronize()
                t0 = time.perf_counter()
                result = model.predict(frame, **options)[0]
                synchronize()
                latency = (time.perf_counter() - t0) * 1000
                timings.append(latency)
                stage_times.append(result.speed)
                boxes = result.boxes.xyxy.cpu().tolist()
                scores = result.boxes.conf.cpu().tolist()
                classes = result.boxes.cls.int().cpu().tolist()
                detections = [dict(class_id=c, label=names[c], confidence=s, xyxy=b)
                              for b, s, c in zip(boxes, scores, classes)]
                raw_detections = detections
                excluded = []
                if osd_filter:
                    detections, excluded = osd_filter.apply(raw_detections)
                for item in excluded:
                    key = item['excluded_by']
                    excluded_counts[key] = excluded_counts.get(key, 0) + 1
                for item in detections:
                    counts[item['label']] += 1
                stream.write(json.dumps(dict(frame=index, source_time_sec=index / fps,
                                             predict_ms=latency, detections=detections,
                                             raw_detections=raw_detections, excluded_detections=excluded)) + '\n')
                if osd_filter:
                    keep = [i for i, d in enumerate(raw_detections) if osd_filter.reason(d) is None]
                    rendered = result[keep].plot() if keep else frame.copy()
                    osd_filter.draw_regions(rendered)
                else:
                    rendered = result.plot()
                caption = f'AURA | YOLOv8m VisDrone | RAW | {actual_device} | {latency:.1f} ms'
                if osd_filter:
                    caption = f'AURA | OSD FILTER ON | excluded {len(excluded)} | {latency:.1f} ms'
                cv2.rectangle(rendered, (0, 0), (width, 26), (0, 0, 0), -1)
                cv2.putText(rendered, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                if writer:
                    writer.write(rendered)
                if processed in [0, 100, 250, 500]:
                    cv2.imwrite(str(args.output / f'frame_{index:06d}.jpg'), rendered)
                processed += 1
                if processed % 100 == 0:
                    print(f'{processed} frames; median prediction {np.median(timings):.1f} ms', flush=True)
                if args.show:
                    cv2.imshow('AURA VisDrone - Q to stop', rendered)
                    if cv2.waitKey(1) & 0xff in [ord('q'), 27]:
                        break
    finally:
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    elapsed = time.perf_counter() - started
    if not timings:
        raise RuntimeError('No frames were processed')
    report = dict(
        model='YOLOv8m VisDrone; CBAM not verified' if attention else 'YOLOv8m VisDrone (no CBAM)',
        weights=str(args.weights.resolve()), weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        attention_modules=attention, classes=names, actual_device=actual_device,
        gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
        versions=dict(torch=torch.__version__, ultralytics=ultralytics.__version__, opencv=cv2.__version__),
        source=str(args.source.resolve()), source_width=width, source_height=height,
        source_fps=fps, source_total_frames=total_frames, processed_frames=processed, stride=args.stride,
        start_frame=args.start_frame,
        imgsz=args.imgsz, confidence=args.conf, iou=args.iou, half=args.device == 'cuda', warmup_frames=5,
        predict_median_ms=float(np.median(timings)), predict_p95_ms=float(np.percentile(timings, 95)),
        predict_mean_ms=float(np.mean(timings)), prediction_throughput_fps=1000 / float(np.mean(timings)),
        offline_loop_fps=processed / elapsed, elapsed_sec=elapsed,
        stage_mean_ms={k:float(np.mean([t[k] for t in stage_times])) for k in stage_times[0]},
        peak_torch_allocated_mb=torch.cuda.max_memory_allocated()/1024**2 if args.device == 'cuda' else None,
        peak_torch_reserved_mb=torch.cuda.max_memory_reserved()/1024**2 if args.device == 'cuda' else None,
        detection_instances_by_class=counts,
        osd_profile=osd_filter.profile if osd_filter else None,
        excluded_instances_by_region=excluded_counts,
        limitations=['Unannotated FPV recording: accuracy, precision, recall and mAP are not measured.',
                     'Detection counts include repeated objects across frames; they are not unique people/vehicles.',
                     'Offline loop FPS includes decode, drawing and saving; it is not live camera-to-screen latency.',
                     'Prediction timing includes preprocessing, inference and postprocessing, with CUDA synchronization.',
                     'PyTorch memory excludes other processes and driver allocations.',
                     'Optional OSD filtering can exclude real targets overlapping configured regions; excluded boxes remain logged.'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
