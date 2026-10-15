"""Interactive recording viewer with three switchable detection modes."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import time

from detection_modes import MODES, render, select_detections
from osd_filter import OsdFilter

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT.parent / 'feeed enhancement/samples/fpv/real_fpv_user_flight.mp4')
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/yolov8m_visdrone_recovered.pt')
    parser.add_argument('--profile', type=Path, default=ROOT / 'profiles/user_flight_osd.json')
    parser.add_argument('--mode', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--title', default='Model Experiment 1 - YOLOv8m VisDrone')
    parser.add_argument('--headless-frames', type=int, default=0, help='Smoke check: cycle all modes without opening a window')
    args = parser.parse_args()
    import cv2
    import torch
    from ultralytics import YOLO
    if not args.weights.is_file() or not args.source.is_file():
        parser.error('Existing weights and source recording are required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; this experiment requires the GPU environment')
    model = YOLO(str(args.weights)).to(0)
    cap = cv2.VideoCapture(str(args.source))
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError('Cannot read recording')
    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError('Invalid recording frame rate')
    osd = OsdFilter(args.profile, width, height)
    options = dict(imgsz=640, conf=.25, iou=.7, device=0, quantize=16, verbose=False, max_det=300)
    for _ in range(5):
        model.predict(frame, **options)
    output = ROOT / 'outputs' / ('interactive_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True)
    (output / 'session.json').write_text(json.dumps(dict(
        title=args.title, weights=str(args.weights.resolve()),
        weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        source=str(args.source.resolve()), profile=osd.profile, classes=model.names,
        inference=options, modes=MODES,
        note='Recording preview; switching affects display only. Exclusions can hide real targets.'
    ), indent=2), encoding='utf-8')
    state = dict(mode=args.mode - 1, paused=False)
    window = 'AURA - Detection experiments'
    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and 0 <= y < 34 and 0 <= x < width:
            state['mode'] = min(2, x * 3 // width)
    if not args.headless_frames:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, mouse)
    index, processed, previous_mode = 0, 0, None
    try:
        with (output / 'events.jsonl').open('w', encoding='utf-8') as log:
            while ok:
                began = time.perf_counter()
                torch.cuda.synchronize()
                started = time.perf_counter()
                result = model.predict(frame, **options)[0]
                torch.cuda.synchronize()
                latency = (time.perf_counter() - started) * 1000
                raw = [dict(class_id=int(c), label=model.names[int(c)], confidence=float(s), xyxy=b)
                       for b, s, c in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())]
                log.write(json.dumps(dict(event='prediction', frame=index, raw_detections=raw, predict_ms=latency)) + '\n')
                restart, stop = False, False
                while True:
                    if args.headless_frames:
                        state['mode'] = processed % 3
                    mode = state['mode']
                    keep, excluded = select_detections(raw, mode, osd)
                    if mode != previous_mode:
                        log.write(json.dumps(dict(event='mode', frame=index, mode=MODES[mode])) + '\n')
                        previous_mode = mode
                    picture = render(frame, result, raw, mode, osd,
                                     f'{args.title} | frame {index} | {latency:.1f} ms', state['paused'])
                    if args.headless_frames:
                        cv2.imwrite(str(output / f'mode_{mode + 1}.jpg'), picture)
                        break
                    cv2.imshow(window, picture)
                    key = cv2.waitKey(15) & 0xff
                    if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        stop = True
                        break
                    if key in (ord('1'), ord('2'), ord('3')):
                        state['mode'] = key - ord('1')
                        continue
                    if key == ord(' '):
                        state['paused'] = not state['paused']
                        continue
                    if key == ord('r'):
                        restart = True
                        break
                    if not state['paused'] and time.perf_counter() - began >= 1 / fps:
                        break
                log.write(json.dumps(dict(event='display', frame=index, mode=MODES[mode],
                                          detections=[raw[i] for i in keep], excluded=excluded)) + '\n')
                log.flush()
                processed += 1
                if stop or (args.headless_frames and processed >= args.headless_frames):
                    break
                if restart:
                    if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                        raise RuntimeError('Cannot restart recording')
                    index = 0
                else:
                    index += 1
                ok, frame = cap.read()
    finally:
        cap.release()
        if not args.headless_frames:
            cv2.destroyAllWindows()
    print(f'Viewer log: {output}')


if __name__ == '__main__':
    main()
