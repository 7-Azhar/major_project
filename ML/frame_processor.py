"""Process one frame; reuse predictions when only review controls change."""
import base64
import time

from detection_modes import select_detections
from osd_filter import OsdFilter


class FrameProcessor:
    def __init__(self, cv2, model, hardware, enhancer_factory, profile):
        self.cv2, self.model, self.hardware = cv2, model, hardware
        self.enhancer_factory, self.profile = enhancer_factory, profile
        self.osd, self.size = None, None
        self.reset()

    def reset(self):
        # Temporal enhancement history must not cross a seek or source change.
        self.enhancer = self.enhancer_key = self.result = None
        self.enhanced = self.original = None
        self.raw_detections = []
        self.enhancement_ms = self.inference_ms = 0
        self.meta = {}

    def jpeg(self, image):
        ok, data = self.cv2.imencode('.jpg', image, [self.cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError('Cannot encode preview')
        return base64.b64encode(data).decode('ascii')

    def process(self, raw, cfg, *, new_frame):
        started = time.perf_counter()
        changed_enhancer = self.enhancer_key != cfg['enhancement']
        if changed_enhancer:
            params = dict(device='cuda' if self.hardware.cuda else 'cpu',
                          fp16=self.hardware.fp16, num_curves=4) if cfg['enhancement'] == 'zerodce' else {}
            self.enhancer = self.enhancer_factory(cfg['enhancement'], **params)
            self.enhancer_key = cfg['enhancement']
        if new_frame or changed_enhancer or self.enhanced is None:
            self.hardware.synchronize()
            tick = time.perf_counter()
            self.enhanced, self.meta = self.enhancer.process(raw.copy())
            self.hardware.synchronize()
            self.enhancement_ms = (time.perf_counter() - tick) * 1000
            if self.enhanced.shape != raw.shape:
                raise RuntimeError('Enhancer changed frame dimensions')
            self.result = None
        if new_frame or self.original is None:
            self.original = self.jpeg(raw)
        h, w = raw.shape[:2]
        if self.size != (w, h):
            self.osd = OsdFilter(self.profile, w, h, scale_to_frame=True)
            self.size = (w, h)

        detections, excluded = [], []
        picture = self.enhanced.copy()
        reused = self.result is not None and cfg['detection'] != 'off'
        if cfg['detection'] != 'off':
            if self.result is None:
                self.hardware.synchronize()
                tick = time.perf_counter()
                self.result = self.model.predict(self.enhanced, imgsz=640, conf=.25, iou=.7,
                    device=self.hardware.predict_device, quantize=16 if self.hardware.fp16 else 32,
                    verbose=False, max_det=300)[0]
                self.hardware.synchronize()
                self.inference_ms = (time.perf_counter() - tick) * 1000
                boxes = self.result.boxes
                self.raw_detections = [dict(label=self.model.names[int(c)], confidence=float(s), xyxy=b)
                    for b, s, c in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist())]
            mode = {'original': 0, 'exclude25': 1, 'exclude50': 2}[cfg['detection']]
            keep, excluded = select_detections(self.raw_detections, mode, self.osd)
            detections = [self.raw_detections[i] for i in keep]
            if keep:
                picture = self.result[keep].plot(line_width=1, font_size=12)
            if mode and cfg['outlines']:
                self.osd.draw_regions(picture)
        output = self.jpeg(picture)
        return dict(width=w, height=h, original=self.original, output=output,
                    enhancement_ms=round(self.enhancement_ms, 1),
                    detection_ms=round(self.inference_ms, 1) if cfg['detection'] != 'off' else 0,
                    processing_ms=round((time.perf_counter() - started) * 1000, 1),
                    inference_reused=reused, detections=detections, excluded=excluded,
                    excluded_count=len(excluded), osd_available=True, enhancement_metadata=self.meta)
