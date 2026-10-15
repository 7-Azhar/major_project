"""Local AURA dashboard: existing enhancement -> original VisDrone detector."""
import argparse
import base64
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import time
import uuid
from urllib.parse import unquote
from video_sources import is_camera, open_source

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'feeed enhancement'))
from detection_modes import select_detections
from osd_filter import OsdFilter

ENHANCEMENTS = {'identity': 'Original / raw', 'clahe': 'CLAHE', 'unsharp': 'Unsharp mask',
                'bilateral': 'Bilateral denoise', 'fpv_restoration': 'FPV restoration',
                'zerodce': 'Zero-DCE', 'fpv_hybrid': 'FPV hybrid',
                'fpv_restoration_v2': 'FPV restoration v2'}
DEFAULT_SOURCE = 'feeed enhancement/samples/fpv/real_fpv_user_flight.mp4'
UPLOADS = ROOT / 'uploads'
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
MAX_UPLOAD = 1024 * 1024 * 1024


class Pipeline:
    def __init__(self, device='auto', precision='auto'):
        import cv2
        import torch
        from ultralytics import YOLO
        from hardware import Hardware
        self.hardware = Hardware(torch, device, precision)
        print('AURA hardware: ' + json.dumps(self.hardware.report()), flush=True)
        self.cv2, self.torch = cv2, torch
        self.model = YOLO(str(ROOT / 'models/yolov8m_visdrone_recovered.pt')).to(self.hardware.device)
        from detect_video import VISDRONE_CLASSES
        if list(self.model.names.values()) != VISDRONE_CLASSES:
            raise ValueError('Unexpected detector class mapping')
        self.sources = [DEFAULT_SOURCE] + sorted({str(p.relative_to(PROJECT)).replace('\\', '/')
                        for p in PROJECT.glob('*') if p.suffix.lower() in ('.mp4', '.mov', '.avi')})
        UPLOADS.mkdir(exist_ok=True)
        self.sources += [p.relative_to(PROJECT).as_posix() for p in UPLOADS.glob('*/*') if p.suffix.lower() in VIDEO_EXTENSIONS]
        self.condition = threading.Condition()
        self.stop = False
        self.config = dict(source=DEFAULT_SOURCE, enhancement='identity', detection='exclude25', playing=False, outlines=True, powered=True)
        self.revision, self.seek = 0, 0
        self.snapshot = dict(status='loading', error=None, config=self.config.copy(), frame=None)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def update(self, change):
        allowed = {'source', 'enhancement', 'detection', 'playing', 'outlines', 'seek', 'powered'}
        if not isinstance(change, dict) or set(change) - allowed:
            raise ValueError('Unknown control')
        if 'source' in change and change['source'] not in self.sources and not is_camera(change['source']):
            raise ValueError('Choose a listed local recording')
        if 'enhancement' in change and change['enhancement'] not in ENHANCEMENTS:
            raise ValueError('Unknown enhancement')
        if 'detection' in change and change['detection'] not in ('off', 'original', 'exclude25', 'exclude50'):
            raise ValueError('Unknown detection mode')
        for key in ('playing', 'outlines', 'powered'):
            if key in change and type(change[key]) is not bool:
                raise ValueError('Expected a boolean')
        if 'seek' in change and (type(change['seek']) is not int or change['seek'] < 0):
            raise ValueError('Seek must be a nonnegative frame number')
        with self.condition:
            if 'seek' in change and is_camera(change.get('source', self.config['source'])):
                raise ValueError('Live cameras cannot seek or restart. Disconnect and reconnect instead.')
            if 'seek' in change:
                total = self.snapshot.get('total_frames', 0)
                if total and change.get('source', self.config['source']) == self.config['source'] and change['seek'] >= total:
                    raise ValueError('Seek is outside this recording')
                self.seek = change['seek']
            if 'source' in change and change['source'] != self.config['source']:
                self.seek = None if is_camera(change['source']) else change.get('seek', 0)
                self.config['playing'] = False
            self.config.update({k: v for k, v in change.items() if k != 'seek'})
            self.revision += 1
            self.condition.notify_all()
            return self.revision

    def run(self):
        from app.enhancers import get_enhancer_by_name
        self.hardware.activate()  # CUDA current device is thread-local.
        cv2, torch = self.cv2, self.torch
        cap, source, enhancer, enhancer_key = None, None, None, None
        raw, enhanced, meta = None, None, {}
        last_revision, index = -1, -1
        try:
            while not self.stop:
                with self.condition:
                    if not self.config['powered']:
                        if is_camera(source) and cap:
                            cap.release()
                            cap, source, raw, enhancer_key = None, None, None, None
                        self.snapshot = dict(status='powered off', error=None, config=self.config.copy(),
                                             frame=index, revision=self.revision, detections=[], excluded_count=0,
                                             enhancement_ms=0, detection_ms=0)
                        self.condition.wait(.2)
                        continue
                    if last_revision == self.revision and not self.config['playing']:
                        self.condition.wait(.2)
                        continue
                    cfg, revision, seek = self.config.copy(), self.revision, self.seek
                    self.seek = None
                started = time.perf_counter()
                try:
                    changed_frame = False
                    live = is_camera(cfg['source'])
                    if source != cfg['source'] or (live and (cap is None or not cap.isOpened()) and cfg['playing']):
                        if cap:
                            cap.release()
                        cap = open_source(cv2, PROJECT, cfg['source'])
                        source = cfg['source']
                        fps = cap.get(cv2.CAP_PROP_FPS) or 25
                        if not math.isfinite(fps) or fps <= 0:
                            fps = 25
                        total = 0 if live else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        raw, enhancer_key, index = None, None, -1
                        seek = None if live else (0 if seek is None else seek)
                    if seek is not None:
                        if seek >= total:
                            raise ValueError('Seek is outside this recording')
                        if not cap.set(cv2.CAP_PROP_POS_FRAMES, seek):
                            raise RuntimeError('Cannot seek recording')
                        index, raw, enhancer_key = seek - 1, None, None
                    if raw is None or (cfg['playing'] and revision == last_revision):
                        ok, candidate = cap.read()
                        if not ok:
                            if live:
                                raise RuntimeError('Camera stopped delivering frames. Check the cable, then reconnect.')
                            with self.condition:
                                self.config['playing'] = False
                                self.snapshot['status'] = 'ended'
                                self.snapshot['config']['playing'] = False
                            last_revision = revision
                            continue
                        raw, index, changed_frame = candidate, index + 1, True
                    changed_enhancer = enhancer_key != cfg['enhancement']
                    if changed_enhancer:
                        enhancer = get_enhancer_by_name(cfg['enhancement'], **({'device': 'cuda' if self.hardware.cuda else 'cpu', 'fp16': self.hardware.fp16, 'num_curves': 4}
                                                       if cfg['enhancement'] == 'zerodce' else {}))
                        enhancer_key = cfg['enhancement']
                    if changed_frame or changed_enhancer:
                        self.hardware.synchronize()
                        tick = time.perf_counter()
                        enhanced, meta = enhancer.process(raw.copy())
                        self.hardware.synchronize()
                        enhancement_ms = (time.perf_counter() - tick) * 1000
                        if enhanced.shape != raw.shape:
                            raise RuntimeError('Enhancer changed frame dimensions')
                    h, w = raw.shape[:2]
                    osd = OsdFilter(ROOT / 'profiles/user_flight_osd.json', w, h, scale_to_frame=True)
                    detections, excluded, detection_ms = [], [], 0
                    picture = enhanced.copy()
                    if cfg['detection'] != 'off':
                        self.hardware.synchronize()
                        tick = time.perf_counter()
                        result = self.model.predict(enhanced, imgsz=640, conf=.25, iou=.7, device=self.hardware.predict_device,
                                                    quantize=16 if self.hardware.fp16 else 32, verbose=False, max_det=300)[0]
                        self.hardware.synchronize()
                        detection_ms = (time.perf_counter() - tick) * 1000
                        raw_detections = [dict(label=self.model.names[int(c)], confidence=float(s), xyxy=b)
                            for b, s, c in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())]
                        mode = {'original': 0, 'exclude25': 1, 'exclude50': 2}[cfg['detection']]
                        keep, excluded = select_detections(raw_detections, mode, osd)
                        detections = [raw_detections[i] for i in keep]
                        if keep:
                            picture = result[keep].plot(line_width=1, font_size=12)
                        if mode and cfg['outlines']:
                            osd.draw_regions(picture)
                    def jpeg(image):
                        ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if not ok:
                            raise RuntimeError('Cannot encode preview')
                        return base64.b64encode(data).decode('ascii')
                    packet = dict(status='playing' if cfg['playing'] else 'paused', error=None, revision=revision,
                        live=live,
                        config=cfg, frame=index, total_frames=total, source_fps=fps, width=w, height=h,
                        original=jpeg(raw), output=jpeg(picture), enhancement_ms=round(enhancement_ms, 1),
                        detection_ms=round(detection_ms, 1), processing_ms=round((time.perf_counter()-started)*1000, 1),
                        detections=detections, excluded_count=len(excluded), osd_available=True,
                        enhancement_metadata=meta, gpu=self.hardware.name, hardware=self.hardware.report())
                    with self.condition:
                        if revision == self.revision:
                            self.snapshot = packet
                    last_revision = revision
                    if live and not cfg['playing']:
                        cap.release()
                    with self.condition:
                        self.condition.wait(max(0, 1/fps - (time.perf_counter()-started)))
                except Exception as exc:
                    if cap:
                        cap.release()
                    source = None
                    with self.condition:
                        if revision == self.revision:
                            self.config['playing'] = False
                            self.snapshot = dict(status='error', error=str(exc), config=self.config.copy(), frame=index, revision=revision)
                    last_revision = revision
        finally:
            if cap:
                cap.release()

    def close(self):
        with self.condition:
            self.stop = True
            self.condition.notify_all()
        self.thread.join(timeout=10)


def serve(port=8765, device='auto', precision='auto'):
    pipeline = Pipeline(device, precision)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, data, mime='application/json', code=200):
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == '/':
                self.send((ROOT / 'dashboard/index.html').read_bytes(), 'text/html; charset=utf-8')
            elif self.path == '/sources.js':
                self.send((ROOT / 'dashboard/sources.js').read_bytes(), 'text/javascript; charset=utf-8')
            elif self.path == '/api/options':
                self.send(json.dumps(dict(sources=pipeline.sources, enhancements=ENHANCEMENTS)).encode())
            elif self.path == '/api/frame':
                with pipeline.condition:
                    packet = pipeline.snapshot.copy()
                self.send(json.dumps(packet).encode())
            else:
                self.send(b'{}', code=404)

        def do_POST(self):
            # Same-origin JSON only; no remote control of the local GPU service.
            if self.headers.get('Origin') not in (None, f'http://127.0.0.1:{port}', f'http://localhost:{port}'):
                self.send(b'{"error":"Origin rejected"}', code=403)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if self.path == '/api/upload':
                    if self.headers.get('Content-Type') != 'application/octet-stream' or not 0 < size <= MAX_UPLOAD:
                        raise ValueError('Choose a video between 1 byte and 1 GB.')
                    name = unquote(self.headers.get('X-Filename', '')).replace('\\', '/').split('/')[-1]
                    suffix = Path(name).suffix.lower()
                    if suffix not in VIDEO_EXTENSIONS:
                        raise ValueError('Supported video containers: MP4, MOV, AVI, MKV, WEBM, M4V.')
                    # The browser filename is display-only; generate a safe storage name.
                    import re
                    name = re.sub(r'[^a-zA-Z0-9_. -]', '_', Path(name).stem)[:100] + suffix
                    folder = UPLOADS / uuid.uuid4().hex
                    folder.mkdir()
                    path = folder / ('video-' + name)
                    try:
                        self.connection.settimeout(60)
                        with path.open('wb') as target:
                            remaining = size
                            while remaining:
                                chunk = self.rfile.read(min(1024 * 1024, remaining))
                                if not chunk:
                                    raise ValueError('Upload interrupted')
                                target.write(chunk)
                                remaining -= len(chunk)
                        probe = pipeline.cv2.VideoCapture(str(path))
                        try:
                            ok, _ = probe.read()
                            if not ok:
                                raise ValueError('This file is not a readable video. Try an MP4 with H.264 video.')
                        finally:
                            probe.release()
                        source = path.relative_to(PROJECT).as_posix()
                        with pipeline.condition:
                            pipeline.sources.append(source)
                        self.send(json.dumps(dict(source=source, name=name)).encode())
                    except Exception:
                        path.unlink(missing_ok=True)
                        folder.rmdir()
                        raise
                    return
                if self.path != '/api/control' or not 0 < size <= 4096 or self.headers.get('Content-Type') != 'application/json':
                    raise ValueError('Expected a small JSON control request')
                revision = pipeline.update(json.loads(self.rfile.read(size)))
                self.send(json.dumps(dict(revision=revision)).encode())
            except (ValueError, TypeError, OSError) as exc:
                self.send(json.dumps(dict(error=str(exc))).encode(), code=400)
    class LocalServer(ThreadingHTTPServer):
        allow_reuse_address = False
    server = LocalServer(('127.0.0.1', port), Handler)
    print(f'AURA dashboard ready: http://127.0.0.1:{port}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        pipeline.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--precision', choices=['auto', 'fp32'], default='auto')
    args = parser.parse_args()
    serve(args.port, args.device, args.precision)
