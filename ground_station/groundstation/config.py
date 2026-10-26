import json
import os
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from dataclasses import asdict
import hashlib

ROOT = Path(__file__).resolve().parent.parent


def config_fingerprint(config):
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


@dataclass
class Config:
    mode: str = "demo"
    video_backend: str = "native"
    aura_port: int = 8766
    video_source: str | int = "demo"
    video_width: int = 960
    video_height: int = 540
    video_fps: int = 30
    jpeg_quality: int = 75
    mavlink_endpoint: str = "udpin:0.0.0.0:14550"
    mavlink_baud: int = 460800
    target_system: int = 1
    target_component: int = 1
    live_control: bool = False
    live_missions: bool = False
    max_speed_mps: float = 0.5
    max_altitude_m: float = 30
    geofence_radius_m: float = 50
    minimum_battery_percent: int = 25
    home_latitude: float = 17.385
    home_longitude: float = 78.4867
    yolo_model: str = ""
    detection_interval_s: float = 0.3
    confidence_threshold: float = 0.5
    ocr_interval_s: float = 1.0
    tesseract_cmd: str = ""
    osd_regions: list = field(default_factory=list)

    def __post_init__(self):
        for name in ('live_control', 'live_missions'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be a JSON boolean, true or false')
        for name in ('video_width', 'video_height', 'video_fps', 'jpeg_quality', 'mavlink_baud',
                     'target_system', 'target_component', 'minimum_battery_percent'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer')
        for name in ('home_latitude','home_longitude','max_speed_mps','max_altitude_m',
                     'geofence_radius_m','detection_interval_s','ocr_interval_s','confidence_threshold'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{name} must be a finite number')
        if type(self.video_source) not in (int, str) or self.video_source == '' or (type(self.video_source) is int and self.video_source < 0):
            raise ValueError('video_source must be a nonnegative camera index or a source string')
        for name in ('mode','video_backend','mavlink_endpoint','yolo_model','tesseract_cmd'):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f'{name} must be text')
        if not self.mavlink_endpoint or self.mavlink_baud <= 0 or not 1 <= self.target_system <= 255 or not 1 <= self.target_component <= 255:
            raise ValueError('Invalid MAVLink endpoint, baud or target IDs')
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be from 1 to 100')
        if not isinstance(self.osd_regions, list):
            raise ValueError('osd_regions must be a list')
        for region in self.osd_regions:
            if not isinstance(region, dict) or not isinstance(region.get('name'), str):
                raise ValueError('Each OSD region needs a name and normalized box')
            box = region.get('box')
            if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
                raise ValueError('OSD boxes require four finite numbers')
            x,y,w,h = box
            if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1-x and 0 < h <= 1-y):
                raise ValueError('OSD box is outside the image')
        if self.video_backend not in ('native', 'aura'):
            raise ValueError('video_backend must be native or aura')
        if type(self.aura_port) is not int or not 1024 <= self.aura_port <= 65535:
            raise ValueError('aura_port must be a local port from 1024 to 65535')
        if not all(math.isfinite(v) for v in (self.home_latitude,self.home_longitude,self.max_speed_mps,self.max_altitude_m,self.geofence_radius_m)):
            raise ValueError('Navigation configuration must be finite')
        if not -85<=self.home_latitude<=85 or not -180<=self.home_longitude<=180:
            raise ValueError('Invalid configured home coordinates')
        if self.mode not in ("demo", "live"):
            raise ValueError("mode must be demo or live")
        if not 0 < self.max_speed_mps <= 2 or not 0 < self.geofence_radius_m <= 500:
            raise ValueError("Invalid control limits")
        if not 0 < self.max_altitude_m <= 120 or not 0 <= self.minimum_battery_percent <= 100:
            raise ValueError("Invalid altitude or battery limit")
        if not 1 <= self.video_fps <= 60 or not 160 <= self.video_width <= 1920 or not 120 <= self.video_height <= 1080:
            raise ValueError("Invalid video settings")
        if self.detection_interval_s < 0.1 or self.ocr_interval_s < 0.2:
            raise ValueError("Processing intervals too short")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Invalid confidence threshold")
        if self.mode == "live" and self.video_source == "demo" and self.video_backend != 'aura':
            raise ValueError("Live mode requires a real video source index or file")

    @classmethod
    def load(cls):
        path = Path(os.environ.get("GS_CONFIG", str(ROOT / "config.json")))
        if 'GS_CONFIG' in os.environ and not path.is_file():
            raise ValueError(f'Configuration file not found: {path}')
        raw = json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
        if not isinstance(raw, dict):
            raise ValueError('Configuration must be a JSON object')
        allowed = {f.name for f in fields(cls)}
        unknown = raw.keys() - allowed
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        return cls(**raw)
