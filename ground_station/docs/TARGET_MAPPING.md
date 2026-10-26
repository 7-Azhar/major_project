# Detection mapping and the future hardware connection

The main page is now an observation workspace: processed video, people/vehicle tracks,
estimated target locations, review, recording and brief operator-approved movement.
The older mission editor remains at /advanced and is not the normal workflow.

## The two laptop inputs

1. **Video:** fixed analog camera → compatible receiver/goggle video output →
   USB capture device → AURA camera selector → enhancement → original VisDrone detector.
   The capture device must expose usable video to OpenCV/Windows. A cable or USB port
   on the goggles does not, by itself, establish video-output compatibility.
2. **Telemetry:** flight controller → configured ELRS MAVLink link → laptop USB serial
   or UDP endpoint → ground station. This is separate from receiving the analog video.
   Confirm the actual flight controller, firmware, ELRS TX/RX and receiver/capture model.

Until this hardware is connected, the default configuration continues video review.
Simulated aircraft coordinates are deliberately withheld from the main detection map.
Recordings never use the current aircraft GPS to invent historical object locations.

## What an object coordinate means

Aircraft GPS identifies the aircraft. A monocular image alone does not provide the
distance to an arbitrary person. Our implemented estimator uses:

- A measured pinhole camera calibration (fx, fy, cx, cy and five OpenCV distortion coefficients).
- Measured fixed-camera mounting roll, pitch and yaw. Positive mount_pitch_down_deg
  means the optical axis points below the aircraft's forward axis.
- Fresh MAVLink position, MSL altitude and ATTITUDE roll/pitch/yaw matched to video time.
- A surveyed flat ground elevation in the **same MSL datum**, with a centre and valid radius.
- Measured video and telemetry delays, plus explicit timing, position, height, angle
  and pixel-error assumptions.

The estimator undistorts the bottom centre of the detection box, rotates the camera ray
into north/east/down, and intersects it with the surveyed plane. It then converts the
north/east offset to latitude/longitude. The box bottom is only a likely ground contact:
hidden feet, roofs, trees, slopes, inaccurate boxes or airborne objects violate that assumption.

The UI labels all coordinates **Estimated**. Circles show a conservative sensitivity
estimate from the configured error assumptions and available GPS accuracy; they are
not statistically calibrated confidence regions or guaranteed bounds. They do not
account for unknown terrain, incorrect lens models or detector mistakes. This is not
an exact-GPS or person-identity system.

No coordinate is published when calibration is unverified or belongs to another camera,
video is recorded/paused/stale, pose is missing or too far from video time, GPS lacks a
3D fix, the ground-contact box is clipped, the ray is near/above the horizon, or the
range, surveyed area or estimated error limit is exceeded.

## Calibration workflow when the drone arrives

Open **Hardware & mapping setup**, download the JSON template, replace every example
with measured values, then import it. The template always has verified=false.
Only set verified=true after checking all measurements against known ground targets.
Unverified calibration is saved but cannot generate real map locations.

- Calibrate from the actual USB-captured video at the same crop and aspect ratio.
  Resizing without cropping is supported; an aspect-ratio change is rejected.
  Changing lens, camera tilt, capture crop or camera index requires recalibration.
  Fisheye lenses need a suitable calibration/rectification step; do not apply these
  pinhole coefficients to an unmodelled fisheye image.
- Survey the relevant ground plane, centre and radius. Do not enter the aircraft's
  altitude above home as ground elevation or assume it is height above a target.
- Measure delay from scene capture to the software capture timestamp and delay from
  telemetry measurement to laptop arrival. Enter video_delay_ms and telemetry_delay_ms.
  pose_skew_ms reports remaining arrival-based sample mismatch. This is not hardware
  clock synchronisation; delayed variable-rate links can still defeat the estimate.
- Use timing_error_ms and other error fields conservatively. Defaults are examples,
  not measured specifications of this drone.
- Compare repeated estimates against known ground points at different ranges/headings.
  Use measured errors to determine usable ranges; do not accept a coordinate just
  because a marker appears.

Calibration is stored locally in ground_station/data/camera-calibration.json. It is
not pushed to Git. Applying a new calibration clears current object tracks to avoid
mixing coordinate systems. The backend validates all fields and rejects partial
verified profiles rather than silently filling unknown measurements.

## Duplicate detection and review

- Highly overlapping same-category boxes are suppressed within each frame.
- Confirmation requires three distinct consecutive detection frames.
- One-to-one association uses image overlap, and nearby ground estimates for short
  reappearances. Repeated polling of one image cannot increase confirmation.
- A short occlusion can reuse a track ID. Two simultaneous separate detections are
  kept separate. Ambiguous matches create a new track instead of silently merging people.
- Source/seek changes, telemetry reconnect/reboot and calibration changes reset tracks.
- Lost objects are shown as last seen, with an age. Session records expire from the
  current map after three minutes; the local event log retains logged observations.
  Unconfirmed candidates disappear after three seconds without another sighting.
- Verify / Dismiss is an operator review action. It does not prove the object's identity
  or improve the detector. A dismissed ongoing track does not create a new alert every frame.

IDs are local tracking labels, not face recognition or permanent re-identification.
Crossing people, large camera movement, heavy occlusion and a return after a long gap
can split or swap tracks. The system cannot guarantee a global unique-person count.

**Export observations** includes unlocated candidates and reasons. **Export map**
includes only confirmed, located, non-dismissed tracks as GeoJSON, with error radius,
review status and last-seen time. object_confirmed/object_update/object_review events
are persisted in the session log. The sample preview is explicitly synthetic and
never enters real calibration, exports, tracking history or flight control.

## Small AI-assisted movements

A persistent person track can suggest a brief hold. The operator must approve it.
Left/right/forward/back requests last 0.5 seconds and remain subject to the existing
GPS, video, battery, altitude, configured radius, pilot arming/mode and short lease gates.
There is no new automatic takeoff, landing, person-following or autonomous search.
The exact distance and ability to hold depend on the aircraft and firmware.

The supplied live configuration still has live_control=false and live_missions=false.
Only the existing ArduCopter GUIDED velocity path is implemented. ELRS/MAVLink support
alone does not imply the unknown flight controller supports it. Confirm firmware and
bench-test reception/control later. Read HARDWARE.md and COMMISSIONING.md first.

For reception later, copy config.live.example.json to config.json, set the actual
MAVLink serial/UDP endpoint, baud rate and aircraft/component IDs, then run:

    .\ground_station\.venv\Scripts\python.exe .\ground_station\launch.py --config .\ground_station\config.json

No aircraft connection or command was used to validate this implementation.

## Technical references

- MAVLink position / MSL versus home-relative altitude:
  https://mavlink.io/en/messages/common.html#GLOBAL_POSITION_INT
- MAVLink aircraft attitude and GPS accuracy fields:
  https://mavlink.io/en/messages/common.html#ATTITUDE
  https://mavlink.io/en/messages/common.html#GPS_RAW_INT
- OpenCV camera calibration and undistortPoints:
  https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- ExpressLRS MAVLink requirements and laptop connections:
  https://www.expresslrs.org/software/mavlink/
