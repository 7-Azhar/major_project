"""Consume AURA results without opening a second camera or loading a model.

Video sequence is a display revision; detection sequence identifies the actual
captured frame so changing a threshold cannot confirm a person three times.
"""
import base64
import json
import math
import threading
import time
from urllib.parse import urlencode
from urllib.request import build_opener, ProxyHandler

import cv2
import numpy as np


class AuraVideo:
    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.frame = self.jpeg = None
        self.frame_at = 0
        self.sequence = 0
        self.packet = {}
        self.connected = False
        self.status = 'Waiting for AURA vision service'
        self.fps = 0
        self.last_frame_key = None
        self.last_arrival = None
        self.demo_person_visible = self.demo_video_loss = False
        self.demo_data = {}
        self.threads = [threading.Thread(target=self.poll, daemon=True, name='aura-bridge')]

    def accept(self, packet, *, now=None, wall=None):
        now = time.monotonic() if now is None else now
        wall = time.time() if wall is None else wall
        cfg = packet.get('config', {})
        active = cfg.get('powered', False) and not packet.get('error') and bool(packet.get('output'))
        jpeg = base64.b64decode(packet['output'], validate=True) if active else None
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) if jpeg else None
        if jpeg and frame is None:
            raise ValueError('AURA returned an invalid image')
        # Old AURA servers have no captured_at: they remain viewable, but cannot
        # supply fresh flight-control or temporal-tracking evidence.
        captured = packet.get('captured_at')
        age = wall-captured if isinstance(captured, (int, float)) and math.isfinite(captured) else None
        fresh_time = now-max(0, age) if age is not None and age >= -.1 and active else 0
        key = (packet.get('server_id'), packet.get('source_epoch'), packet.get('frame'))
        with self.lock:
            if key != self.last_frame_key and fresh_time:
                if self.last_arrival is not None and now > self.last_arrival:
                    self.fps = 1/(now-self.last_arrival)
                self.last_arrival, self.last_frame_key = now, key
            self.frame, self.jpeg, self.frame_at = frame, jpeg, fresh_time
            self.packet, self.connected = packet, True
            self.sequence += 1
            self.status = 'AURA · ' + (packet.get('error') or packet.get('status', 'waiting'))

    def disconnect(self, message):
        with self.lock:
            self.connected = False
            self.status = 'AURA disconnected · ' + message
            self.frame_at = 0
            self.jpeg = self.frame = None

    def poll(self):
        opener = build_opener(ProxyHandler({}))
        after, server = None, None
        while not self.stop_event.is_set():
            query = '?' + urlencode({'after': after, 'server_id': server}) if after is not None else ''
            try:
                with opener.open(f'http://127.0.0.1:{self.config.aura_port}/api/frame' + query, timeout=5) as response:
                    if response.status == 204:
                        continue
                    packet = json.load(response)
                self.accept(packet)
                after, server = packet['sequence'], packet['server_id']
            except Exception as exc:
                self.disconnect(str(exc))
                after = server = None
                self.stop_event.wait(1)

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            p, cfg = self.packet, self.packet.get('config', {})
            age = max(0, now-self.frame_at) if self.frame_at and self.connected else None
            playing = cfg.get('powered') and cfg.get('playing') and p.get('status') == 'playing'
            fresh = playing and age is not None and age < 1 and not self.demo_video_loss
            w, h = p.get('width', 0), p.get('height', 0)
            detections = []
            if self.connected and self.jpeg and w > 0 and h > 0:
                for detection in p.get('detections', []):
                    values = detection.get('xyxy', [])
                    confidence = detection.get('confidence')
                    if len(values) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                        continue
                    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                        continue
                    original = detection['label']
                    # Keep the model's exact class alongside the shared person category.
                    label = 'person' if original in ('pedestrian', 'people', 'person') else original
                    box = [max(0., min(1., float(v)/size)) for v, size in zip(detection['xyxy'], (w,h,w,h))]
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    detections.append(dict(label=label, original_label=original, box=box,
                        confidence=detection['confidence'], detector='AURA YOLOv8m VisDrone'))
            source_id = f"{p.get('server_id')}:{p.get('source_epoch')}:{cfg.get('source')}"
            return dict(status=self.status, age_s=age if playing and not self.demo_video_loss else None,
                captured_monotonic=self.frame_at or None, image_size=[w,h],
                sequence=self.sequence, fps=round(self.fps, 1) if fresh else 0,
                processing_ms=p.get('processing_ms', 0), inference_ms=p.get('detection_ms', 0),
                inference_frame_age_s=age, ai_status='AURA enhancement → VisDrone detection' if self.connected else self.status,
                ocr_status='Not enabled in AURA bridge; overlay exclusion is not OCR',
                detections=detections if fresh else [], review_detections=detections,
                detection_sequence=f"{source_id}:{p.get('frame')}", source_id=source_id,
                source_frame=p.get('frame'), review_available=bool(self.connected and self.jpeg),
                osd=[], source=cfg.get('source', ''), live_ready=bool(fresh and p.get('live') and str(cfg.get('source', '')).startswith('camera:')),
                annotations_embedded=True, backend='aura', connected=self.connected,
                aura=dict(config=cfg, status=p.get('status'), excluded_count=p.get('excluded_count', 0),
                          enhancement_ms=p.get('enhancement_ms', 0), hardware=p.get('hardware', {})))

    def close(self):
        self.stop_event.set()
        for thread in self.threads:
            if thread.is_alive():
                thread.join(6)
