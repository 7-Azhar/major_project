import math
import re
import time

DIRECTIONS = {
    "forward": (1, 0, 0), "backward": (-1, 0, 0),
    "left": (0, -1, 0), "right": (0, 1, 0),
    "up": (0, 0, -1), "down": (0, 0, 1), "hold": (0, 0, 0),
}


def parse_command(text):
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    cleaned = re.sub(r"^(move|go) ", "", cleaned)
    aliases = {"back": "backward", "stop": "hold", "hover": "hold"}
    cleaned = aliases.get(cleaned, cleaned)
    if cleaned not in DIRECTIONS:
        raise ValueError("Use forward, backward, left, right, up, down, or hold")
    return cleaned


def distance_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2-p1, math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 6371000 * 2 * math.asin(min(1, math.sqrt(a)))


def control_blockers(config, telemetry, enabled, frame_age, now=None):
    if config.mode == "demo":
        return []
    now = time.monotonic() if now is None else now
    reasons = []
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)
    if not config.live_control:
        reasons.append("Live control disabled in config")
    if not enabled:
        reasons.append("Operator control lease not enabled")
    for name in ("heartbeat", "position", "gps", "battery", "ekf"):
        stamp = telemetry.get('timestamps', {}).get(name)
        if not finite(stamp) or not 0 <= now-stamp <= 2.5:
            reasons.append(f"{name} telemetry stale or missing")
    if telemetry.get("mode") != "GUIDED":
        reasons.append("Pilot must select GUIDED")
    if not telemetry.get("armed"):
        reasons.append("Drone not armed by pilot")
    if telemetry.get("fix_type", 0) < 3:
        reasons.append("3D GPS fix required")
    if not telemetry.get("ekf_ok"):
        reasons.append("EKF position estimate not healthy")
    if not finite(telemetry.get('battery_percent')) or not config.minimum_battery_percent <= telemetry['battery_percent'] <= 100:
        reasons.append("Battery low or unknown")
    altitude = telemetry.get("altitude_m")
    if not finite(altitude) or not 1 <= altitude <= config.max_altitude_m:
        reasons.append("Altitude outside control envelope")
    lat, lon = telemetry.get("latitude"), telemetry.get("longitude")
    if not finite(lat) or not finite(lon) or not -85 <= lat <= 85 or not -180 <= lon <= 180 or distance_m(config.home_latitude, config.home_longitude, lat, lon) > config.geofence_radius_m:
        reasons.append("Outside configured home radius")
    if not finite(frame_age) or not 0 <= frame_age <= 1:
        reasons.append("Video stale or unavailable")
    return reasons


def search_grid(lat, lon, width, height, spacing):
    if not -85 <= lat <= 85 or not -180 <= lon <= 180:
        raise ValueError("Invalid search origin")
    if not 5 <= width <= 200 or not 5 <= height <= 200 or not 2 <= spacing <= 50:
        raise ValueError("Use dimensions 5–200 m and spacing 2–50 m")
    rows = math.ceil(height / spacing)
    points = []
    for row in range(rows + 1):
        y = min(row * spacing, height)
        for x in ([0, width] if row % 2 == 0 else [width, 0]):
            points.append({"latitude": lat + y / 111320,
                           "longitude": lon + x / (111320 * math.cos(math.radians(lat)))})
    return points
