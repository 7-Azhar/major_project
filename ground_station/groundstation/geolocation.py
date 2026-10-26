"""Fixed-camera, flat-ground estimates. Never substitute aircraft GPS for target GPS."""
from dataclasses import asdict, dataclass, field, fields
import math

import cv2
import numpy as np


@dataclass
class Calibration:
    verified: bool = False
    source: str = "camera:1"
    image_width: int = 640
    image_height: int = 480
    fx: float = 500
    fy: float = 500
    cx: float = 320
    cy: float = 240
    distortion: list = field(default_factory=lambda: [0, 0, 0, 0, 0])
    mount_pitch_down_deg: float = 0
    mount_roll_deg: float = 0
    mount_yaw_deg: float = 0
    ground_elevation_msl_m: float = 0
    ground_reference_latitude: float = 0
    ground_reference_longitude: float = 0
    ground_valid_radius_m: float = 100
    video_delay_ms: float = 0
    telemetry_delay_ms: float = 0
    timing_error_ms: float = 100
    max_pose_skew_ms: float = 200
    gps_error_m: float = 5
    height_error_m: float = 3
    angle_error_deg: float = 3
    pixel_error_px: float = 8
    max_range_m: float = 150
    max_error_m: float = 30

    def __post_init__(self):
        if type(self.verified) is not bool:
            raise ValueError("verified must be a JSON boolean")
        if not isinstance(self.source, str) or not self.source.startswith('camera:') or not self.source[7:].isdigit():
            raise ValueError("Bind calibration to a camera source, such as camera:1")
        if any(type(v) is not int or not 120 <= v <= 4096 for v in (self.image_width, self.image_height)):
            raise ValueError("Invalid calibration image size")
        for key, value in asdict(self).items():
            if key not in ('verified', 'source', 'distortion') and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError(f"{key} must be a finite number")
        if not isinstance(self.distortion, list) or len(self.distortion) != 5 or any(
                type(v) not in (int,float) or not math.isfinite(v) or abs(v)>10 for v in self.distortion):
            raise ValueError("Use five measured OpenCV distortion coefficients")
        if not (10 <= self.fx <= 10000 and 10 <= self.fy <= 10000 and
                0 <= self.cx < self.image_width and 0 <= self.cy < self.image_height):
            raise ValueError("Invalid camera intrinsics")
        bounds = {
            'mount_pitch_down_deg':(-90,90), 'mount_roll_deg':(-180,180), 'mount_yaw_deg':(-180,180),
            'ground_elevation_msl_m':(-500,9000), 'ground_reference_latitude':(-85,85),
            'ground_reference_longitude':(-180,180), 'ground_valid_radius_m':(1,500),
            'video_delay_ms':(0,3000), 'telemetry_delay_ms':(0,3000), 'timing_error_ms':(10,1000),
            'max_pose_skew_ms':(20,500), 'gps_error_m':(1,100), 'height_error_m':(.5,100),
            'angle_error_deg':(.5,20), 'pixel_error_px':(1,100), 'max_range_m':(5,500),
            'max_error_m':(1,100)}
        for key, (low,high) in bounds.items():
            if not low <= getattr(self,key) <= high:
                raise ValueError(f"{key} must be between {low} and {high}")

    @classmethod
    def parse(cls, data):
        if not isinstance(data,dict) or data.keys()-{f.name for f in fields(cls)}:
            raise ValueError("Unknown calibration fields; use the supplied template")
        if data.get('verified') is True and set(data) != {f.name for f in fields(cls)}:
            raise ValueError("A verified calibration must explicitly supply every field")
        return cls(**data)


def rotation(roll, pitch, yaw):
    r,p,y = map(math.radians,(roll,pitch,yaw))
    cr,sr,cp,sp,cy,sy = math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def ground_offset(calibration, pose, u, v, height):
    c = calibration
    matrix = np.array([[c.fx,0,c.cx],[0,c.fy,c.cy],[0,0,1]],dtype=float)
    xy = cv2.undistortPoints(np.array([[[u,v]]],dtype=float),matrix,np.array(c.distortion,dtype=float))[0,0]
    # Camera right/down/forward -> body forward/right/down, then body -> north/east/down.
    body = rotation(c.mount_roll_deg,-c.mount_pitch_down_deg,c.mount_yaw_deg) @ np.array([1,xy[0],xy[1]])
    ray = rotation(pose['roll_deg'],pose['pitch_deg'],pose['yaw_deg']) @ body
    if not np.isfinite(ray).all() or ray[2]/np.linalg.norm(ray) < math.sin(math.radians(10)):
        raise ValueError("Ground ray is above or too close to the horizon")
    offset = ray[:2] * height/ray[2]
    if np.linalg.norm(offset) > c.max_range_m:
        raise ValueError("Target is beyond the calibrated mapping range")
    return offset


def locate(box, pose, calibration, image_size):
    c = calibration
    if not c.verified:
        raise ValueError("Camera and ground calibration not verified")
    w,h = image_size
    if w <= 0 or h <= 0 or abs((w/h)/(c.image_width/c.image_height)-1) > .01:
        raise ValueError("Video aspect ratio does not match camera calibration")
    if len(box)!=4 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in box):
        raise ValueError("Invalid object box")
    x,y,x2,y2 = box
    if not (0 < x < x2 < 1 and 0 < y < y2 < .985):
        raise ValueError("Object touches the image edge; ground contact may be hidden")
    required=('latitude','longitude','altitude_msl_m','roll_deg','pitch_deg','yaw_deg')
    if any(type(pose.get(k)) not in (int,float) or not math.isfinite(pose[k]) for k in required):
        raise ValueError("Complete GPS, MSL altitude and camera attitude are required")
    if not -85 <= pose['latitude'] <= 85 or not -180 <= pose['longitude'] <= 180:
        raise ValueError("Invalid aircraft GPS")
    height = pose['altitude_msl_m']-c.ground_elevation_msl_m
    height_error = max(c.height_error_m,pose.get('vertical_accuracy_m') or 0)
    if not max(1,height_error) < height <= 500:
        raise ValueError("Height above the surveyed ground is unavailable or too uncertain")
    u,v = (x+x2)*.5*c.image_width,y2*c.image_height
    nominal = ground_offset(c,pose,u,v,height)
    timing = (c.timing_error_ms+pose.get('pose_skew_ms',0))/1000
    angle = c.angle_error_deg+abs(pose.get('angular_rate_dps',0))*timing
    if angle > 20:
        raise ValueError("Camera rotation or timing uncertainty is too high")
    deviations=[]
    for key in ('roll_deg','pitch_deg','yaw_deg'):
        deviations.append(max(np.linalg.norm(ground_offset(c,{**pose,key:pose[key]+sign*angle},u,v,height)-nominal) for sign in (-1,1)))
    deviations.append(max(np.linalg.norm(ground_offset(c,pose,u,v,height+sign*height_error)-nominal) for sign in (-1,1)))
    pixel=max(c.pixel_error_px,(y2-y)*c.image_height*.05)
    deviations.append(max(np.linalg.norm(ground_offset(c,pose,u+du,v+dv,height)-nominal)
                          for du,dv in ((pixel,0),(-pixel,0),(0,pixel),(0,-pixel))))
    error = max(c.gps_error_m,pose.get('horizontal_accuracy_m') or 0)+sum(deviations)+abs(pose.get('speed_mps') or 0)*timing
    if not math.isfinite(error) or error>c.max_error_m:
        raise ValueError("Estimated location uncertainty exceeds the mapping limit")
    lat = pose['latitude']+math.degrees(nominal[0]/6378137)
    lon = pose['longitude']+math.degrees(nominal[1]/(6378137*math.cos(math.radians(pose['latitude']))))
    from .control import distance_m
    if distance_m(c.ground_reference_latitude,c.ground_reference_longitude,lat,lon)+error>c.ground_valid_radius_m:
        raise ValueError("Target is outside the surveyed flat-ground area")
    return dict(latitude=lat,longitude=lon,error_radius_m=round(error,1),
                range_m=round(float(np.linalg.norm(nominal)),1),method='fixed-camera / surveyed flat ground',
                estimated=True, timing='arrival-aligned with configured delay',
                pose_skew_ms=pose.get('pose_skew_ms'),ground_contact='bounding-box bottom centre')
