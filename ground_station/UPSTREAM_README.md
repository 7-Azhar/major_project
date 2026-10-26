# Aero Ground Station

A laptop-based FPV observation and supervised control project for analogue video, person detection, GPS telemetry and rescue incident review. Built for Windows / Python 3.11+. The software runs locally; internet is not required for the default interface or detector.

## Run now

From this project folder in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

Open http://127.0.0.1:8000. Default mode is **demo**: generated video, generated GPS/flight telemetry and simulated movement. No radio link is opened in demo mode. The Mission Planner-style interface places flight instruments on the left, an interactive map and waypoint table in the centre, and mission/AI actions on the right.

Try this complete AI mission demonstration:

1. Generate `search 20 by 20 meters at 10 meters` in Mission copilot.
2. Click **Use proposed route**, then **Validate & apply route**. Drag map markers or edit the table before validation if needed.
3. Open **Simulation lab**, choose 10× simulation rate and apply the scenario.
4. Start the mission and watch the aircraft follow the route using GPS feedback.
5. Select **Auto-pause mission on person**. In Simulation lab, enable **Person enters scene** and apply. After three distinct matched detection frames, the mission pauses and an incident is recorded.
6. Clear the person scenario and explicitly resume. Test GPS loss, video loss or low battery; each pauses the mission and requires an operator resume after recovery.

The plain-language copilot is a bounded local intent planner, not a language model. It supports search/survey, rectangular perimeter, polygonal orbit, measured north/south/east/west/body-relative/up/down movements, and return home. Example requests: `search 20 by 20 meters at 10 meters`, `perimeter 15 by 15 meters`, `orbit 10 meters at 12 meters`, `move north 5 meters`, `return home`. Unknown requests and routes outside the envelope are rejected. HOG/optional YOLO provide actual vision inference on USB/file footage. Demo detections are synthetic annotations and cannot demonstrate real detection accuracy.

Dependencies have been installed in `.venv`. For a fresh installation:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pytest -q
```

The startup script bypass flag applies to that invocation only. It does not change your machine's execution-policy setting. You can also run the server directly:

```powershell
.\.venv\Scripts\python.exe -m uvicorn groundstation.main:app --host 127.0.0.1 --port 8000
```

## What is implemented

- Mission Planner-style responsive workspace with a pan/zoom map, click-to-add/drag-to-move waypoints, editable/reorderable waypoint table and relative altitude/hold settings.
- Offline tactical map and optional Esri satellite / OpenStreetMap street tiles, selected by the operator. Online layers send requested map areas to the provider; no tile prefetching or offline bulk download is implemented. Required attribution stays on the map.
- Live MJPEG video, attitude instrument, telemetry freshness, temperature (barometer telemetry where available), battery, flight mode, GPS coordinates and aircraft trail.
- USB/file capture via OpenCV; capture, inference and telemetry operate independently. The application retains the newest frame instead of queueing inference jobs.
- Built-in OpenCV HOG person detector. Optional local YOLO model for richer object detection. Boxes state their source frame; HOG scores are not represented as probabilities.
- Optional Tesseract OSD text extraction from configured regions. OCR stays separate from flight telemetry and never authorizes movement.
- Temporal person tracking in image coordinates; three matched distinct frames establish a persistent candidate. Optional policy automatically pauses an active mission and records a rescue lead, while advisory mode leaves it to the operator. These remain candidates, not verified people or victim coordinates.
- Bounded local mission copilot, validated lawnmower routes, GPS-feedback waypoint execution, progress, start/pause/resume/cancel, return-home route planning, SQLite mission save/load, JSON/GeoJSON import and JSON/QGC WPL export. WPL imports support relative-altitude WAYPOINTs only and reject unsupported flight commands. No takeoff is inserted into exports.
- Simulation lab with person injection, GPS/video loss, battery threshold scenarios and accelerated mission time. Safety pauses stay latched after recovery.
- Bidirectional MAVLink adapter for serial or UDP. Reads heartbeat, GPS, position, battery, radio status, EKF and flight messages; sends bounded body-frame velocity requests in ArduCopter GUIDED.
- Controls for forward/back/left/right/up/down/hold; strict typed command grammar; optional browser dictation with explicit review before Send.
- Laptop control lease, command expiry, per-message freshness gates, configured radius/altitude/speed limits and pilot mode-change cancellation.
- SQLite event history, aircraft telemetry samples, incident snapshots and JSON log export. Bounded asynchronous 10 fps MJPEG AVI recording, up to 5 minutes or 500 MB per clip. Search lawnmower route preview / GeoJSON export.
- API documentation at `/docs`, test suite, reproducible base dependency lock and hardware commissioning instructions.

## What has NOT been validated

The capture card has not been bought and the drone/radio/GPS are not connected to this project. Hardware capture, over-air MAVLink, command response, RF latency, manual takeover and real flight are therefore unverified. Automated tests and demo operation are software verification only.

This version does not implement autonomous takeoff/arming/landing, onboard MAVLink mission upload/download, obstacle avoidance, person following, exact victim geolocation, multivehicle control, or production WebRTC. Guided waypoint execution is implemented in the laptop controller and validated with the software simulator; its physical flight behaviour and ArduPilot SITL integration remain unverified. Flight stabilization and loss-of-link failsafes must remain onboard.

## Connect your hardware

Read `docs/HARDWARE.md` and `docs/COMMISSIONING.md` first. Create `config.json` by copying `config.example.json`; set `mode` to `live`, `video_source` to the capture-device index, and configure the MAVLink endpoint. Leave `live_control` and `live_missions` false while verifying reception. Guided mission execution requires both flags, fresh heading/position/video and the operator's three-second control lease. It sends bounded velocity setpoints and does not arm, take off, switch mode or upload an onboard mission.

```powershell
.\.venv\Scripts\python.exe .\scripts\devices.py
```

For a RadioMaster Pocket internal ELRS module, use compatible **TX Backpack Wi-Fi MAVLink forwarding** and `udpin:0.0.0.0:14550`. The handset USB joystick connection is not a MAVLink telemetry connection. A compatible external transmitter module may support serial MAVLink over its own USB port. Confirm your exact hardware.

The API binds to loopback only. Same-origin and random per-process write-token checks protect browser actions. Do not publish or expose this control server to a public network. This is a single-operator application, not a cloud fleet platform. UDP MAVLink is not authenticated here; use a trusted dedicated network and configure the vehicle's failsafes separately.

## AI / OSD upgrades

The default HOG detector requires no extra download, but is a baseline for upright nearby people; aerial FPV angles, small objects, blur and OSD overlays are challenging. Validate real footage and choose an appropriate trained model before treating detections as useful rescue leads.

For optional YOLO and OCR:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ai.txt
```

Place an appropriately licensed YOLO detection model in `models/`, set `yolo_model` to its existing file path, and restart. Models are not downloaded automatically. Review Ultralytics/model licensing for your intended distribution. Tesseract is a separate executable; set `tesseract_cmd` to its installed path. Example OSD region, calibrated to YOUR actual captured image:

```json
"osd_regions": [{"name":"battery", "box":[0.02,0.88,0.24,0.10]}]
```

Each box is normalized `[x, y, width, height]`. The reader outputs unverified raw text, not reliable navigation measurements. OSD temperature is only available if your flight controller/video system actually displays it; the project cannot infer a sensor that is not present.

## Latency

The display shows frame age since the OpenCV read started and software read/encode time. These are **not camera-to-screen end-to-end latency**. Capture drivers, analogue receiver, USB buffers, JPEG delivery and browser rendering add delay. Measure the real system using a visible stopwatch/LED and direct-view comparison. `CAP_PROP_BUFFERSIZE` is a request that some drivers ignore. MJPEG is a local prototype transport; advanced transport and measured deadlines are future work.

## Project layout

`groundstation/` contains configuration, video, telemetry, decision rules, control gates, logging and FastAPI routes. `web/` contains the browser interface. `tests/` validates control gating, protocol encoding, simulator behaviour, API protection and incident storage. `data/` is generated locally and contains potentially sensitive images and GPS history; manage retention manually for now.

## Primary references

- [Basement Creations video, downloaded setup/CAD files and project adaptation notes](docs/VIDEO_REFERENCE.md)

- [ExpressLRS MAVLink setup and hardware constraints](https://www.expresslrs.org/software/mavlink/)
- [ArduCopter Guided command frames and masks](https://ardupilot.org/dev/docs/copter-commands-in-guided-mode.html)
- [SpeedyBee F405 AIO support, UART mapping and OSD](https://ardupilot.org/copter/docs/common-speedybeef405aio.html)
- [OpenCV capture API](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)
- [Ultralytics prediction API](https://docs.ultralytics.com/modes/predict/)
