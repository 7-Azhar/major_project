import os
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from collections import deque

# Avoid oversubscribing the laptop across capture and inference threads.
cv2.setNumThreads(2)


class Video:
    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.frame = None
        self.jpeg = None
        self.frame_at = 0
        self.sequence = 0
        self.detections = []
        self.detection_sequence = None
        self.detection_at = 0
        self.osd = []
        self.status = "Starting video"
        self.ai_status = "Starting built-in HOG person detector"
        self.ocr_status = "Disabled — configure OSD regions"
        self.processing_ms = 0
        self.actual_fps = 0
        self.demo_person_visible = False
        self.demo_video_loss = False
        self.inference_ms = 0
        self.inference_frame_age_s = None
        self.frame_times = deque(maxlen=60)
        self.demo_data = {}
        self.threads = [threading.Thread(target=self.capture, daemon=True),
                        threading.Thread(target=self.analyze, daemon=True)]

    def demo_frame(self, t):
        h, w = self.config.video_height, self.config.video_width
        f = np.zeros((h, w, 3), np.uint8)
        f[:] = (42, 53, 38)
        for x in range(0, w, 65):
            cv2.line(f, (x, 0), (x + 100, h), (53, 69, 48), 20)
        cx = int(w*.5 + w*.2*np.sin(t/4))
        if self.demo_person_visible:
            cv2.rectangle(f, (cx-15, h//2), (cx+15, h//2+75), (60, 130, 220), -1)
            cv2.circle(f, (cx, h//2-12), 13, (130, 170, 215), -1)
        cv2.putText(f, "SIMULATED VIDEO / NO AIRCRAFT CONNECTED", (20, 35), 0, .65, (240, 240, 240), 2)
        t=self.demo_data
        osd=f"{t.get('voltage_v',15.6):.1f}V   GPS {t.get('satellites',14)}   ALT {t.get('altitude_m',10):.1f}m"
        cv2.putText(f, osd, (20, h-20), 0, .65, (240, 240, 240), 2)
        return f

    def capture(self):
        cap = None
        file_input = False
        playback_fps = self.config.video_fps
        tick = time.monotonic()
        try:
            if self.config.video_source != "demo":
                source = self.config.video_source
                if isinstance(source, str) and source.isdigit():
                    source = int(source)
                backend = cv2.CAP_DSHOW if os.name == "nt" and isinstance(source, int) else cv2.CAP_ANY
                cap = cv2.VideoCapture(source, backend)
                file_input = isinstance(source, str) and Path(source).is_file()
                if file_input:
                    measured = cap.get(cv2.CAP_PROP_FPS)
                    playback_fps = measured if math.isfinite(measured) and measured > 0 else playback_fps
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.video_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.video_height)
                    cap.set(cv2.CAP_PROP_FPS, self.config.video_fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    raise RuntimeError("Capture device cannot open; check index and driver")
            while not self.stop_event.is_set():
                start = time.monotonic()
                if cap is None:
                    if self.demo_video_loss:
                        self.status='Simulated video loss'
                        self.stop_event.wait(.1)
                        continue
                    frame = self.demo_frame(start)
                else:
                    ok, frame = cap.read()
                    if not ok:
                        if file_input:
                            self.status = 'Recording ended'
                            break
                        self.status = "No frames from capture device"
                        self.stop_event.wait(.1)
                        continue
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
                if ok:
                    with self.lock:
                        self.frame = frame
                        self.jpeg = encoded.tobytes()
                        self.frame_at = start
                        self.sequence += 1
                        self.processing_ms = (time.monotonic()-start)*1000
                        self.frame_times.append(start)
                        self.actual_fps = ((len(self.frame_times)-1)/max(.001,start-self.frame_times[0])) if len(self.frame_times)>1 else 0
                        self.status = "Simulated video" if cap is None else "USB/file capture active"
                    tick = start
                if cap is None or file_input:
                    self.stop_event.wait(max(0, 1/playback_fps-(time.monotonic()-start)))
        except Exception as exc:
            self.status = f"Video error: {exc}"
        finally:
            if cap:
                cap.release()

    def analyze(self):
        model = None
        ocr = None
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.ai_status = "Built-in HOG person detector ready (baseline)"
        if self.config.yolo_model:
            try:
                from ultralytics import YOLO
                path = Path(self.config.yolo_model)
                if not path.is_file():
                    raise ValueError("Model must be an existing local file")
                model = YOLO(str(path))
                self.ai_status = "YOLO ready"
            except Exception as exc:
                self.ai_status = f"AI unavailable: {exc}"
        if self.config.osd_regions:
            try:
                import pytesseract
                if self.config.tesseract_cmd:
                    pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd
                pytesseract.get_tesseract_version()
                ocr = pytesseract
                self.ocr_status = "OSD reader ready (untrusted text)"
            except Exception as exc:
                self.ocr_status = f"OCR unavailable: {exc}"
        last_ai = last_ocr = 0
        last_sequence = -1
        while not self.stop_event.wait(.05):
            with self.lock:
                frame = None if self.frame is None else self.frame.copy()
                sequence, frame_at = self.sequence, self.frame_at
            if frame is None or time.monotonic()-frame_at > 1:
                continue
            if sequence==last_sequence:
                continue
            last_sequence=sequence
            now = time.monotonic()
            if self.config.video_source == "demo":
                # Generated annotations are not evidence of detection accuracy.
                h, w = frame.shape[:2]
                cx = int(w*.5 + w*.2*np.sin(frame_at/4))
                with self.lock:
                    self.detections = [{"label": "person", "confidence": None,
                                        "detector": "SIMULATED annotation",
                                        "box": [(cx-20)/w, (h/2-28)/h, (cx+20)/w, (h/2+80)/h]}] if self.demo_person_visible else []
                    self.detection_sequence, self.detection_at = sequence, frame_at
                self.ai_status = "Simulated annotations · real detector runs on USB/file footage"
                continue
            if model and now-last_ai >= self.config.detection_interval_s:
                last_ai = now
                try:
                    result = model.predict(frame, conf=self.config.confidence_threshold, imgsz=640, verbose=False)[0]
                    self.inference_ms=(time.monotonic()-now)*1000
                    self.inference_frame_age_s=time.monotonic()-frame_at
                    h, w = frame.shape[:2]
                    boxes = [{"label": result.names[int(b.cls.item())], "confidence": float(b.conf.item()),
                              "box": [float(x)/size for x, size in zip(b.xyxy[0].tolist(), (w, h, w, h))]}
                             for b in result.boxes]
                    with self.lock:
                        self.detections, self.detection_sequence = boxes, sequence
                        self.detection_at = frame_at
                    self.ai_status = "YOLO running"
                except Exception as exc:
                    self.ai_status = f"Inference error: {exc}"
            elif not model and now-last_ai >= max(.5, self.config.detection_interval_s):
                last_ai = now
                try:
                    h, w = frame.shape[:2]
                    resized = cv2.resize(frame, (min(640, w), max(128, int(h*min(640, w)/w))))
                    boxes, weights = hog.detectMultiScale(resized, winStride=(8, 8), padding=(8, 8), scale=1.05)
                    self.inference_ms=(time.monotonic()-now)*1000
                    self.inference_frame_age_s=time.monotonic()-frame_at
                    rh, rw = resized.shape[:2]
                    detections = [{"label": "person", "confidence": None,
                                   "score": float(score), "detector": "HOG baseline",
                                   "box": [x/rw, y/rh, (x+bw)/rw, (y+bh)/rh]}
                                  for (x, y, bw, bh), score in zip(boxes, weights) if float(score) > .5]
                    with self.lock:
                        self.detections, self.detection_sequence = detections, sequence
                        self.detection_at = frame_at
                    self.ai_status = "HOG person detector running · use YOLO for better detection"
                except Exception as exc:
                    self.ai_status = f"Baseline detector error: {exc}"
            if ocr and now-last_ocr >= self.config.ocr_interval_s:
                last_ocr = now
                try:
                    readings = []
                    h, w = frame.shape[:2]
                    for region in self.config.osd_regions:
                        x, y, rw, rh = region["box"]
                        if not (0 <= x < 1 and 0 <= y < 1 and 0 < rw <= 1-x and 0 < rh <= 1-y):
                            raise ValueError("OSD boxes must be normalized x,y,width,height within frame")
                        crop = frame[int(y*h):int((y+rh)*h), int(x*w):int((x+rw)*w)]
                        gray = cv2.cvtColor(cv2.resize(crop, None, fx=3, fy=3), cv2.COLOR_BGR2GRAY)
                        text = ocr.image_to_string(gray, config="--psm 7", timeout=2).strip()
                        readings.append({"name": region["name"], "text": text,
                                         "source": "OSD OCR — unverified", "sequence": sequence})
                    with self.lock:
                        self.osd = readings
                except Exception as exc:
                    self.ocr_status = f"OCR error: {exc}"

    def snapshot(self):
        with self.lock:
            age = time.monotonic()-self.frame_at if self.frame_at else None
            source=self.config.video_source
            camera=type(source) is int or isinstance(source,str) and source.isdigit()
            return {"status": self.status, "age_s": age, "sequence": self.sequence,
                    "captured_monotonic":self.detection_at or None,
                    "image_size":[self.frame.shape[1],self.frame.shape[0]] if self.frame is not None else [0,0],
                    "backend":"native", "simulated":source=='demo',
                    "live_ready":bool(camera and age is not None and 0 <= age < 1),
                    "fps": round(self.actual_fps, 1), "processing_ms": round(self.processing_ms, 1),
                    "inference_ms":round(self.inference_ms,1),"inference_frame_age_s":self.inference_frame_age_s,
                    "ai_status": self.ai_status, "ocr_status": self.ocr_status,
                    "detections": self.detections if self.detection_at and time.monotonic()-self.detection_at < 1 else [],
                    "detection_sequence": self.detection_sequence, "osd": self.osd,
                    "source": str(self.config.video_source)}

    def close(self):
        self.stop_event.set()
        for thread in self.threads:
            if thread.is_alive():
                thread.join(3)
