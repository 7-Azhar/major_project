"""Validated mission plans and GPS-feedback navigation in Guided mode."""
import math
import re
from dataclasses import asdict, dataclass

from .control import distance_m, search_grid


@dataclass
class Waypoint:
    latitude: float
    longitude: float
    altitude_m: float = 10
    hold_s: float = 0


def validate_plan(points, config):
    if not 1 <= len(points) <= 100:
        raise ValueError("A mission needs 1–100 waypoints")
    result = []
    for i, point in enumerate(points):
        wp = Waypoint(**point)
        if not all(math.isfinite(v) for v in asdict(wp).values()):
            raise ValueError(f"Waypoint {i+1}: all values must be finite")
        if not -85 <= wp.latitude <= 85 or not -180 <= wp.longitude <= 180:
            raise ValueError(f"Waypoint {i+1}: invalid coordinates")
        if not 1 <= wp.altitude_m <= config.max_altitude_m or not 0 <= wp.hold_s <= 60:
            raise ValueError(f"Waypoint {i+1}: altitude or hold outside limits")
        if distance_m(config.home_latitude, config.home_longitude, wp.latitude, wp.longitude) > config.geofence_radius_m:
            raise ValueError(f"Waypoint {i+1}: outside the {config.geofence_radius_m:g} m configured fence")
        result.append(asdict(wp))
    return result


def route_stats(points, origin=None, speed=1):
    route = ([origin] if origin else []) + points
    length = sum(distance_m(a['latitude'], a['longitude'], b['latitude'], b['longitude'])
                 for a, b in zip(route, route[1:]))
    return {'distance_m': round(length, 1), 'estimated_seconds': round(length/max(.1, speed)+sum(p.get('hold_s', 0) for p in points)),
            'waypoints': len(points)}


def navigator_velocity(telemetry, waypoint, speed):
    lat, lon = telemetry['latitude'], telemetry['longitude']
    north = (waypoint['latitude']-lat)*111320
    east = (waypoint['longitude']-lon)*111320*math.cos(math.radians(lat))
    up = waypoint['altitude_m']-telemetry['altitude_m']
    horizontal = math.hypot(north, east)
    if horizontal < .6 and abs(up) < .4:
        return (0, 0, 0), True, horizontal
    norm = math.sqrt(north*north+east*east+up*up)
    scale = min(speed, norm*.8)/max(.001, norm)
    n, e, down = north*scale, east*scale, -up*scale
    heading = math.radians(telemetry.get('heading_deg') or 0)
    return (n*math.cos(heading)+e*math.sin(heading), -n*math.sin(heading)+e*math.cos(heading), down), False, horizontal


class Mission:
    def __init__(self):
        self.name = 'Untitled search'
        self.points = []
        self.status = 'empty'
        self.index = 0
        self.reason = 'Create or load a route'
        self.hold_started = None
        self.distance_to_target = None
        self.revision = 0

    def load(self, name, points):
        if self.status == 'running':
            raise ValueError('Pause mission before editing the route')
        self.name, self.points = name, points
        self.status, self.index, self.hold_started = 'ready', 0, None
        self.distance_to_target = None
        self.reason = 'Route validated; awaiting start'
        self.revision += 1

    def pause(self, reason):
        if self.status == 'running':
            self.status, self.reason, self.hold_started = 'paused', reason, None

    def start(self, resume=False):
        if not self.points:
            raise ValueError('Create a mission first')
        if self.status == 'running':
            raise ValueError('Mission is already running; pause before restarting')
        if resume and self.status != 'paused':
            raise ValueError('Only a paused mission can resume')
        if not resume:
            self.index = 0
        self.status, self.reason, self.hold_started = 'running', 'Following GPS waypoints', None

    def tick(self, telemetry, speed, now):
        if self.status != 'running':
            return (0, 0, 0)
        velocity, reached, self.distance_to_target = navigator_velocity(telemetry, self.points[self.index], speed)
        if reached:
            if self.hold_started is None:
                self.hold_started = now
            if now-self.hold_started >= self.points[self.index]['hold_s']:
                self.index += 1
                self.hold_started = None
                if self.index >= len(self.points):
                    self.status, self.reason = 'completed', 'All waypoints reached; holding position'
        else:
            # A hold must be continuous inside the waypoint tolerance.
            self.hold_started = None
        return velocity

    def snapshot(self, speed):
        return {'name': self.name, 'points': list(self.points), 'status': self.status, 'index': self.index,
                'reason': self.reason, 'revision': self.revision, 'distance_to_target_m': self.distance_to_target,
                'progress_percent': round(100*self.index/max(1,len(self.points))), **route_stats(self.points, speed=speed)}


def copilot_plan(text, config, origin):
    """Bounded local intent grammar, not an LLM and never an executable prompt."""
    text = re.sub(r'\s+', ' ', text.lower().strip())
    if text in ('return home', 'return to home'):
        points = [asdict(Waypoint(config.home_latitude,config.home_longitude,origin.get('altitude_m') or 10))]
        return {'name': 'Return to configured home', 'points': validate_plan(points,config),
                'explanation': 'Return at the current relative altitude; no automatic landing.'}
    movement=re.fullmatch(r'move (north|south|east|west|forward|backward|left|right|up|down) (\d+(?:\.\d+)?)(?: meters?| m)',text)
    if movement:
        amount=float(movement[2])
        if not .5<=amount<=20:
            raise ValueError('Planned movement must be 0.5–20 meters')
        direction=movement[1];n=e=up=0
        cardinal={'north':(1,0),'south':(-1,0),'east':(0,1),'west':(0,-1)}
        if direction in cardinal:
            n,e=(v*amount for v in cardinal[direction])
        elif direction in ('up','down'):
            up=amount if direction=='up' else -amount
        else:
            heading=origin.get('heading_deg')
            if heading is None:
                raise ValueError('Heading required for body-relative movement planning')
            angle=math.radians(heading+{'forward':0,'right':90,'backward':180,'left':-90}[direction])
            n,e=amount*math.cos(angle),amount*math.sin(angle)
        wp=asdict(Waypoint(origin['latitude']+n/111320,
            origin['longitude']+e/(111320*math.cos(math.radians(origin['latitude']))),
            origin['altitude_m']+up))
        return {'name':f'{direction.title()} {amount:g} m','points':validate_plan([wp],config),
                'explanation':f'GPS-feedback movement proposal: {amount:g} meters {direction}. Review the target before starting.'}
    orbit=re.fullmatch(r'orbit (\d+(?:\.\d+)?)(?: meters?| m)(?: at (\d+(?:\.\d+)?)(?: meters?| m))?',text)
    if orbit:
        radius=float(orbit[1]);altitude=float(orbit[2] or 10)
        if not 3<=radius<=30:
            raise ValueError('Orbit radius must be 3–30 meters')
        points=[]
        for i in range(13):
            angle=2*math.pi*i/12
            points.append(asdict(Waypoint(origin['latitude']+radius*math.cos(angle)/111320,
                origin['longitude']+radius*math.sin(angle)/(111320*math.cos(math.radians(origin['latitude']))),altitude)))
        return {'name':f'{radius:g} m orbit','points':validate_plan(points,config),
                'explanation':'Twelve-sided orbit around the current aircraft location. No camera pointing or obstacle avoidance is inferred.'}
    perimeter=re.fullmatch(r'(?:perimeter|inspect) (\d+(?:\.\d+)?) (?:by|x) (\d+(?:\.\d+)?)(?: meters?| m)?(?: at (\d+(?:\.\d+)?)(?: meters?| m)?)?',text)
    if perimeter:
        width,height=float(perimeter[1]),float(perimeter[2]);altitude=float(perimeter[3] or 10)
        if not 5<=width<=100 or not 5<=height<=100:
            raise ValueError('Perimeter dimensions must be 5–100 meters')
        points=[asdict(Waypoint(origin['latitude']+n/111320,origin['longitude']+e/(111320*math.cos(math.radians(origin['latitude']))),altitude))
                for n,e in ((0,0),(0,width),(height,width),(height,0),(0,0))]
        return {'name':f'{width:g} × {height:g} m perimeter','points':validate_plan(points,config),
                'explanation':'Closed rectangular perimeter from the current aircraft position. Review the terrain and flight envelope.'}
    pattern = r'(?:search|survey) (\d+(?:\.\d+)?) (?:by|x) (\d+(?:\.\d+)?)(?: meters?| m)?(?: at (\d+(?:\.\d+)?)(?: meters?| m)?(?: altitude)?)?'
    match = re.fullmatch(pattern,text)
    if not match:
        raise ValueError('Try search 20 by 20 meters at 10 meters, perimeter 15 by 15 meters, orbit 10 meters, move north 5 meters, or return home.')
    width,height = float(match[1]),float(match[2])
    altitude = float(match[3] or 10)
    spacing = min(10, max(2,width/4))
    points = [{**p,'altitude_m':altitude,'hold_s':0} for p in search_grid(origin['latitude'],origin['longitude'],width,height,spacing)]
    return {'name':f'{width:g} × {height:g} m search', 'points':validate_plan(points,config),
            'explanation':f'Local planner generated alternating search lanes {spacing:g} m apart. Review the route and flight envelope before starting.'}
