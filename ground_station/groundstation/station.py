import threading
import time
import uuid
import math
from pathlib import Path

from .config import ROOT
from .control import DIRECTIONS, control_blockers
from .missions import Mission, copilot_plan, validate_plan
from .intelligence import Intelligence
from .decisions import recommend
from .recording import Recorder
from .storage import Store
from .telemetry import Telemetry
from .video import Video
from .video_evidence import capture_evidence
from .observations import Observations


class Station:
    def __init__(self, config, data_root=None):
        self.config = config
        self.server_id = uuid.uuid4().hex
        self.store = Store(data_root or ROOT / "data")
        self.telemetry = Telemetry(config)
        if config.video_backend == 'aura':
            from .aura import AuraVideo
            self.video = AuraVideo(config)
        else:
            self.video = Video(config)
        self.recorder = Recorder(self.video, self.store)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.enabled_until = 0
        self.command_until = 0
        self.velocity = (0, 0, 0)
        self.action = "Observing"
        self.last_error = ""
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.path = []
        self.last_logged_telemetry = 0
        self.mission = Mission()
        self.intelligence = Intelligence()
        self.observations = Observations(self.store)
        self.simulation = {'person_visible': False, 'gps_loss':False, 'video_loss':False,
                           'battery_percent':84, 'time_scale':1}
        self.last_mission_log_index = -1
        self.simulation_clock = 0

    def mission_blockers(self):
        t,v=self.telemetry.snapshot(),self.video.snapshot()
        if self.config.mode == 'live':
            reasons=self.blockers()
            if not self.config.live_missions:
                reasons.append('Guided mission execution disabled in config')
            if type(t.get('heading_deg')) not in (int, float) or not math.isfinite(t['heading_deg']):
                reasons.append('Heading required for GPS route execution')
            return reasons
        reasons=[]
        if v.get('age_s') is None or v['age_s']>1 or self.simulation['video_loss']:
            reasons.append('Video unavailable')
        if t['fix_type']<3 or self.simulation['gps_loss']:
            reasons.append('GPS unavailable')
        if self.simulation['battery_percent']<self.config.minimum_battery_percent:
            reasons.append('Battery below mission threshold')
        return reasons

    def set_mission(self,name,points):
        with self.lock:
            validated=validate_plan(points,self.config)
            self.mission.load(name,validated)
            self.last_mission_log_index = -1
            self.command_until=0
            self.action='Route validated · awaiting start'
            self.store.add('mission_plan',{'name':name,'waypoints':len(points)})
            return self.mission.snapshot(self.config.max_speed_mps)

    def mission_action(self,action):
        with self.lock:
            if action in ('start','resume'):
                blockers=self.mission_blockers()
                if blockers:
                    raise ValueError('; '.join(blockers))
                self.mission.start(action=='resume')
            elif action=='pause':
                self.mission.pause('Paused by operator')
            elif action=='cancel':
                self.mission.status='cancelled'
                self.mission.reason='Cancelled by operator'
            else:
                raise ValueError('Use start, pause, resume or cancel')
            self.command_until=0
            self.action={'start':'Mission starting','resume':'Mission resuming',
                         'pause':'Mission paused','cancel':'Mission cancelled'}[action]
            self.store.add('mission_'+action,{'name':self.mission.name,'mode':self.config.mode})
            return self.mission.snapshot(self.config.max_speed_mps)

    def copilot(self,text):
        with self.lock:
            t=self.telemetry.snapshot()
            if t['latitude'] is None or t['longitude'] is None or time.monotonic()-t['timestamps'].get('position',0)>2.5:
                raise ValueError('Fresh aircraft position required to plan')
            return copilot_plan(text,self.config,t)

    def start(self):
        self.telemetry.thread.start()
        for thread in self.video.threads:
            thread.start()
        self.thread.start()
        self.store.add("session", {"mode": self.config.mode, "video_source": self.config.video_source})

    def blockers(self):
        video = self.video.snapshot()
        reasons = control_blockers(self.config, self.telemetry.snapshot(),
            time.monotonic() < self.enabled_until, video['age_s'])
        if self.config.mode == 'live' and self.config.video_backend == 'aura' and not video.get('live_ready'):
            reasons.append('AURA needs a fresh, playing live camera; recordings cannot enable flight control')
        if self.config.mode == 'live' and self.config.video_backend == 'native':
            source = self.config.video_source
            if not (type(source) is int or isinstance(source, str) and source.isdigit()):
                reasons.append('A live capture device is required; recordings cannot enable flight control')
        return reasons

    def lease(self, enable):
        with self.lock:
            if enable and self.config.mode == "live" and not self.config.live_control:
                raise ValueError("Enable live_control in config only after bench validation")
            self.enabled_until = time.monotonic()+3 if enable else 0
            if not enable:
                self.command_until = 0
                self.mission.pause('Laptop lease released')
            return {"enabled": enable}

    def command(self, direction, duration=0.5):
        if direction not in DIRECTIONS:
            raise ValueError("Unknown movement")
        if not .1 <= duration <= 1:
            raise ValueError("Movement duration must be 0.1–1 second")
        with self.lock:
            reasons = self.blockers()
            if reasons:
                raise ValueError("; ".join(reasons))
            self.mission.pause('Manual movement requested')
            vz = DIRECTIONS[direction][2]
            altitude = self.telemetry.snapshot().get("altitude_m") or 0
            if direction != "hold" and self.config.mode == "live":
                if not 1 <= altitude-vz*self.config.max_speed_mps*duration <= self.config.max_altitude_m:
                    raise ValueError("Requested movement exceeds altitude envelope")
            self.velocity = tuple(v*self.config.max_speed_mps for v in DIRECTIONS[direction])
            self.command_until = time.monotonic()+duration
            self.action = f"{direction.title()} · {duration:.1f}s"
            self.store.add("command", {"direction": direction, "duration_s": duration,
                                      "mode": self.config.mode, "status": "requested, not aircraft acknowledgement"})
            return {"requested": direction, "duration_s": duration, "mode": self.config.mode}

    def release(self):
        with self.lock:
            self.enabled_until = 0
            self.command_until = 0
            self.mission.pause('Pilot takeover / laptop released')
            self.action = "Pilot control / laptop lease released"
            self.store.add("control_release", {"mode": self.config.mode})

    def incident(self, note):
        now = time.monotonic()
        evidence = capture_evidence(self.video)
        video, jpeg, age = evidence['state'], evidence['jpeg'], evidence['captured_age_s']
        review = video.get('backend') == 'aura' and video.get('review_available')
        t = self.telemetry.snapshot()
        position_fresh = now-t["timestamps"].get("position", 0) < 2.5 and t["fix_type"] >= 3
        payload = {"note": note, "mode": self.config.mode,
                   "drone_position": {"latitude": t["latitude"], "longitude": t["longitude"],
                                      "altitude_m": t["altitude_m"]} if position_fresh else None,
                   "location_meaning": "Aircraft location, NOT person/object coordinates",
                   "frame_sequence": evidence['sequence'], "frame_age_s": age,
                   "source": video.get('source'), "source_id": evidence['source_id'],
                   "source_frame": video.get('source_frame'),
                   "video_context": 'live camera' if video.get('live_ready') else 'video review',
                   "telemetry_context": 'simulated' if self.config.mode == 'demo' else 'live aircraft at note time',
                   "vision_settings": video.get('aura', {}).get('config'),
                   "detection_sequence": video.get('detection_sequence'),
                   "detections_match_frame": evidence['detections_match_frame'],
                   "detections": video.get('review_detections', video['detections']) if review else video['detections']}
        with self.lock:
            payload['object_locations']=[item for item in self.observations.snapshot()['items']
                if evidence['detections_match_frame'] and item['confirmed'] and item['location'] and item['review']!='dismissed'
                and item.get('source')==video.get('source')
                and item.get('source_id')==evidence['source_id']
                and item.get('frame')==video.get('source_frame',evidence['sequence'])]
        if jpeg and (review or (age is not None and age < 1)):
            filename = f"snapshot-{time.time_ns()}.jpg"
            (self.store.root / filename).write_bytes(jpeg)
            payload["snapshot"] = filename
        event_id = self.store.add("incident", payload)
        return {"id": event_id, **payload}

    def state(self):
        t = self.telemetry.snapshot()
        now = time.monotonic()
        stamps = t.pop("timestamps")
        t["ages_s"] = {k: round(now-v, 2) for k, v in stamps.items()}
        with self.lock:
            video = self.video.snapshot()
            return {"mode": self.config.mode, "server_id": self.server_id, "telemetry": t, "video": video,
                    "recording": self.recorder.snapshot(),
                    "observations":self.observations.snapshot(),
                    "mission":self.mission.snapshot(self.config.max_speed_mps),
                    "mission_blockers":self.mission_blockers(),
                    "intelligence":self.intelligence.snapshot(),
                    "simulation":dict(self.simulation) if self.config.mode=='demo' else None,
                    "home":{'latitude':self.config.home_latitude,'longitude':self.config.home_longitude},
                    "recommendation": recommend(t, video, self.config.mode),
                    "action": self.action, "control_blockers": self.blockers(),
                    "lease_active": now < self.enabled_until, "last_error": self.last_error,
                    "path": list(self.path), "limits": {"speed_mps": self.config.max_speed_mps,
                    "altitude_m": self.config.max_altitude_m, "radius_m": self.config.geofence_radius_m}}

    def loop(self):
        previous = time.monotonic()
        sending = False
        while not self.stop_event.wait(.1):
            now = time.monotonic()
            with self.lock:
                self.simulation_clock += min(.2,now-previous)*self.simulation['time_scale']
                v=self.video.snapshot()
                self.observations.update(v,self.telemetry,now)
                self.intelligence.update(v,now)
                if self.mission.status=='running':
                    reasons=self.mission_blockers()
                    if reasons:
                        self.mission.pause('; '.join(reasons))
                        self.store.add('mission_safety_pause',{'reason':self.mission.reason})
                    elif self.intelligence.policy=='pause_on_person' and self.intelligence.snapshot(now)['confirmed_count']:
                        self.mission.pause('AI paused: persistent person candidate; verify rescue lead')
                        if now-self.intelligence.last_auto_incident>30:
                            self.intelligence.last_auto_incident=now
                            self.incident('AI lead: persistent person candidate; aircraft location only')
                        self.store.add('ai_pause',{'reason':self.mission.reason,'mode':self.config.mode})
                blockers = self.blockers()
                active = now < self.command_until and not blockers
                velocity = self.velocity if active else (0, 0, 0)
                mission_active=self.mission.status=='running'
                if mission_active:
                    clock=self.simulation_clock if self.config.mode=='demo' else now
                    velocity=self.mission.tick(self.telemetry.snapshot(),self.config.max_speed_mps,clock)
                    active=True
                    self.action=f'Mission · waypoint {min(self.mission.index+1,len(self.mission.points))}/{len(self.mission.points)}'
                    if self.mission.status == 'completed':
                        self.action = 'Mission completed · holding position'
                        self.store.add('mission_completed', {'name':self.mission.name})
                    if self.mission.index!=self.last_mission_log_index:
                        self.last_mission_log_index=self.mission.index
                        self.store.add('mission_progress',{'index':self.mission.index,'status':self.mission.status})
                if self.config.mode == "demo":
                    factor=self.simulation['time_scale'] if mission_active else 1
                    self.telemetry.demo_tick(velocity, min(.2, now-previous)*factor,self.simulation)
                    self.video.demo_data=self.telemetry.snapshot()
                else:
                    try:
                        if active:
                            self.telemetry.send_velocity(*velocity)
                            sending = True
                        elif sending:
                            # Stop only while the vehicle remains fresh and GUIDED;
                            # never fight a pilot mode change. FC timeout is the final failsafe.
                            t = self.telemetry.snapshot()
                            if t["mode"] == "GUIDED" and now-t["timestamps"].get("heartbeat", 0) < 2.5:
                                self.telemetry.send_velocity(0, 0, 0)
                            sending = False
                    except Exception as exc:
                        self.command_until = self.enabled_until = 0
                        self.mission.pause('Transport failure')
                        self.last_error = f"Control transport failure: {exc}"
                        sending = False
                        self.store.add("control_error", {"error": self.last_error})
                    if blockers and self.command_until > now:
                        self.command_until = 0
                        self.action = "Movement stopped: " + "; ".join(blockers)
                    if self.telemetry.snapshot()["mode"] != "GUIDED":
                        self.enabled_until = 0
                        self.mission.pause('Pilot left GUIDED')
                if not active and self.command_until:
                    self.command_until = 0
                    self.action = "Observing · movement lease expired"
                t = self.telemetry.snapshot()
                if now-self.last_logged_telemetry >= 1:
                    self.last_logged_telemetry = now
                    self.store.add("telemetry", {k: v for k, v in t.items() if k != "timestamps"})
                    if t["latitude"] is not None and now-t["timestamps"].get("position", 0) < 2.5:
                        self.path.append({"latitude": t["latitude"], "longitude": t["longitude"]})
                        self.path = self.path[-300:]
            previous = now

    def close(self):
        self.release()
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(3)
        self.recorder.stop()
        self.video.close()
        self.telemetry.close()
        self.store.close()
