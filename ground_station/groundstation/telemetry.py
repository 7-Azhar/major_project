import math
import threading
import time
from collections import deque

from pymavlink import mavutil


class Telemetry:
    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.connection = None
        self.history = {key:deque(maxlen=600) for key in ('position','attitude','gps')}
        self.last_boot_ms = {}
        self.pose_epoch = 0
        self.status = "Simulated telemetry" if config.mode == "demo" else "Waiting for MAVLink"
        self.values = {"source": config.mode, "mode": "GUIDED" if config.mode == "demo" else "UNKNOWN",
                       "armed": config.mode == "demo", "timestamps": {}, "latitude": None,
                       "longitude": None, "altitude_m": None, "speed_mps": None,
                       "battery_percent": None, "voltage_v": None, "fix_type": 0,
                       "satellites": None, "heading_deg": 0, "ekf_ok": False}
        self.thread = threading.Thread(target=self.run, daemon=True)

    def snapshot(self):
        with self.lock:
            return {**self.values, "timestamps": dict(self.values["timestamps"]), "status": self.status}

    def accept(self, message):
        # Never merge another aircraft's telemetry into this vehicle's control state.
        if message.get_srcSystem() != self.config.target_system or message.get_srcComponent() != self.config.target_component:
            return
        kind = message.get_type()
        now = time.monotonic()
        with self.lock:
            v = self.values
            if kind in ('GLOBAL_POSITION_INT','ATTITUDE'):
                boot = getattr(message,'time_boot_ms',None)
                previous = self.last_boot_ms.get(kind)
                if type(boot) is int and previous is not None and boot < previous-1000:
                    for history in self.history.values(): history.clear()
                    self.last_boot_ms.clear()
                    v['timestamps'].clear()
                    self.pose_epoch += 1
                elif type(boot) is int and previous is not None and boot <= previous:
                    return
                if type(boot) is int: self.last_boot_ms[kind]=boot
            if kind == "HEARTBEAT" and message.type != mavutil.mavlink.MAV_TYPE_GCS:
                v.update(mode=mavutil.mode_string_v10(message), armed=bool(message.base_mode & 128))
                v["timestamps"]["heartbeat"] = now
            elif kind == "GLOBAL_POSITION_INT":
                v.update(latitude=message.lat/1e7, longitude=message.lon/1e7,
                         altitude_m=message.relative_alt/1000,
                         altitude_msl_m=message.alt/1000,
                         speed_mps=math.hypot(message.vx, message.vy)/100,
                         heading_deg=message.hdg/100 if message.hdg != 65535 else None)
                v["timestamps"]["position"] = now
                self.history['position'].append((now,{k:v[k] for k in ('latitude','longitude','altitude_msl_m','speed_mps')}))
            elif kind == "GPS_RAW_INT":
                v.update(fix_type=message.fix_type, satellites=message.satellites_visible)
                v["timestamps"]["gps"] = now
                def accuracy(name):
                    value=getattr(message,name,0)
                    return value/1000 if type(value) in (int,float) and 0<value<4294967295 else None
                v.update(horizontal_accuracy_m=accuracy('h_acc'),vertical_accuracy_m=accuracy('v_acc'))
                self.history['gps'].append((now,{k:v[k] for k in ('fix_type','horizontal_accuracy_m','vertical_accuracy_m')}))
            elif kind == "SYS_STATUS":
                v.update(battery_percent=message.battery_remaining if message.battery_remaining >= 0 else None,
                         voltage_v=message.voltage_battery/1000 if message.voltage_battery != 65535 else None)
                v["timestamps"]["battery"] = now
            elif kind == "EKF_STATUS_REPORT":
                # Attitude, horizontal velocity, absolute horizontal position, vertical position;
                # reject constant-position and uninitialized estimates.
                v["ekf_ok"] = (message.flags & 23) == 23 and not bool(message.flags & (128 | 1024))
                v["timestamps"]["ekf"] = now
            elif kind == "RADIO_STATUS":
                v.update(radio_rssi=message.rssi, radio_remrssi=message.remrssi,
                         tx_buffer_percent=message.txbuf)
                v["timestamps"]["radio"] = now
            elif kind == "STATUSTEXT":
                v["flight_message"] = message.text
            elif kind == "ATTITUDE":
                v.update(roll_deg=math.degrees(message.roll),pitch_deg=math.degrees(message.pitch),yaw_deg=math.degrees(message.yaw))
                v['timestamps']['attitude'] = now
                rates=[getattr(message,k,0) for k in ('rollspeed','pitchspeed','yawspeed')]
                rates=[abs(math.degrees(x)) for x in rates if type(x) in (int,float) and math.isfinite(x)]
                self.history['attitude'].append((now,{**{k:v[k] for k in ('roll_deg','pitch_deg','yaw_deg')},
                                                     'angular_rate_dps':max(rates,default=0)}))
            elif kind == "SCALED_PRESSURE":
                v['temperature_c'] = message.temperature/100
                v['timestamps']['temperature'] = now
            self.status = "MAVLink receiving"

    def pose_at(self, captured_at, calibration):
        """Match independent telemetry streams to video time; never use demo or home-relative height."""
        if self.config.mode!='live':
            raise ValueError('Real MAVLink telemetry required; simulated GPS is not object GPS')
        if type(captured_at) not in (int,float) or not math.isfinite(captured_at):
            raise ValueError('Video capture timestamp unavailable')
        target=captured_at-calibration.video_delay_ms/1000
        offset=calibration.telemetry_delay_ms/1000
        pose={}
        skew=0
        with self.lock:
            for key in ('position','attitude','gps'):
                if not self.history[key]:
                    raise ValueError(f'MAVLink {key} is unavailable')
                stamp,values=min(self.history[key],key=lambda item:abs(item[0]-offset-target))
                delta=abs(stamp-offset-target)
                limit=2.5 if key=='gps' else calibration.max_pose_skew_ms/1000
                if delta>limit:
                    raise ValueError(f'MAVLink {key} does not match the video time')
                if key!='gps': skew=max(skew,delta)
                pose.update(values)
            pose['pose_epoch']=self.pose_epoch
        if pose.get('fix_type',0)<3:
            raise ValueError('A valid 3D GPS fix is required')
        pose['pose_skew_ms']=round(skew*1000,1)
        return pose

    def send_velocity(self, vx, vy, vz):
        with self.lock:
            if self.connection is None:
                raise RuntimeError("MAVLink transport unavailable")
            # BODY_NED: forward/right/down; only velocity fields are active.
            self.connection.mav.set_position_target_local_ned_send(
                0, self.config.target_system, self.config.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED, 0x0DC7,
                0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)

    def demo_tick(self, velocity, dt, simulation=None):
        with self.lock:
            v = self.values
            simulation = simulation or {}
            heading=math.radians(v.get('heading_deg') or 0)
            north=velocity[0]*math.cos(heading)-velocity[1]*math.sin(heading)
            east=velocity[0]*math.sin(heading)+velocity[1]*math.cos(heading)
            v["latitude"] = (v["latitude"] if v["latitude"] is not None else self.config.home_latitude) + north*dt/111320
            v["longitude"] = (v["longitude"] if v["longitude"] is not None else self.config.home_longitude) + east*dt/(111320*math.cos(math.radians(v["latitude"])))
            v.update(altitude_m=max(0, (v["altitude_m"] if v["altitude_m"] is not None else 10)-velocity[2]*dt),
                     speed_mps=math.hypot(*velocity[:2]), battery_percent=simulation.get('battery_percent',84), voltage_v=15.6,
                     fix_type=0 if simulation.get('gps_loss') else 3, satellites=0 if simulation.get('gps_loss') else 14, ekf_ok=True,
                     roll_deg=velocity[1]*8,pitch_deg=-velocity[0]*8,temperature_c=36)
            v["timestamps"] = {k: time.monotonic() for k in ("heartbeat", "position", "gps", "battery", "ekf")}

    def run(self):
        if self.config.mode == "demo":
            return
        while not self.stop_event.is_set():
            try:
                conn = mavutil.mavlink_connection(self.config.mavlink_endpoint,
                    baud=self.config.mavlink_baud, source_system=255, source_component=190,
                    autoreconnect=True)
                # Keep ESP-based external TX modules out of reset on Windows.
                if hasattr(conn, "port") and hasattr(conn.port, "dtr"):
                    conn.port.dtr = False
                    conn.port.rts = False
                with self.lock:
                    self.connection = conn
                last_heartbeat = 0
                while not self.stop_event.is_set():
                    with self.lock:
                        msg = conn.recv_match(blocking=False)
                        if time.monotonic()-last_heartbeat > 1:
                            conn.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                            last_heartbeat = time.monotonic()
                    if msg:
                        self.accept(msg)
                    else:
                        self.stop_event.wait(0.01)
            except Exception as exc:
                self.status = f"MAVLink error: {exc}"
            finally:
                with self.lock:
                    if self.connection:
                        self.connection.close()
                    self.connection = None
                    for history in self.history.values(): history.clear()
                    self.last_boot_ms.clear()
                    self.values['timestamps'].clear()
                    self.pose_epoch += 1
            self.stop_event.wait(2)

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(3)
