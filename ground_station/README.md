# AURA + Aero Ground Station

The main workspace focuses on AURA video, detected people/vehicles, estimated object locations and simple operator-approved movements. Large route planning is kept at /advanced. Camera feeds, USB capture devices and recorded videos are supported. FPV is one input use case; its overlay exclusion can be turned off in Vision controls.

**Object mapping:** stable session track IDs, duplicate-box suppression, consecutive-frame confirmation, brief-occlusion matching, last-seen markers, review/dismiss actions and GeoJSON export are implemented. Geographic estimates require a calibrated fixed camera, real time-aligned MAVLink pose and surveyed ground. With current recordings and no aircraft, detections remain unlocated. The sample map is explicitly synthetic. Read the [mapping/hardware guide](docs/TARGET_MAPPING.md) before calibration.

## What was understood and integrated

Your friend's original code is cloned separately in `external/aero-ground-station/` (ignored by the parent repository). The integration lives here, so the original clone and its Git history stay untouched.

| Component | Source | Integration |
| --- | --- | --- |
| Video selection, upload, cameras, eight enhancements, VisDrone detector, exclusions, GPU selection | AURA | One existing AURA service; no second camera reader or model copy |
| Map, waypoints, local mission planner, telemetry, simulated missions | Aero | Retained in ground station |
| Person tracking, incident snapshots, recording | Aero + adapter | Consume AURA's processed frames and filtered detections |
| MAVLink reception and supervised movement | Aero | Existing gates retained; recorded, paused, stale or powered-off AURA video cannot satisfy live video requirements |
| UI | Both | Detection map, Vision controls dialog, separate vision website and vision power at top right; previous mission workspace under Advanced |

The upstream project includes real MAVLink transport code and simulated mission execution. Its own README says physical hardware and real flight were not validated. This integration does not turn those software tests into hardware verification. The local mission copilot is a bounded intent parser, not a language model.

## Windows setup

1. Install Python 3.11 64-bit and complete **Setup AURA.cmd** if this is a fresh laptop.
2. Run **Setup Ground Station.cmd** once. It installs only the ground station's packages into `ground_station/.venv`. It does not change AURA's Torch/CUDA environment.
3. Run **Run Both Workspaces.bat** to open both the detection workspace (`http://127.0.0.1:8000/`) and standalone vision (`http://127.0.0.1:8766/`). **Run Vision Dashboard.bat** opens vision alone. The older **Start Ground Station.cmd** opens only the detection page while starting its shared pipeline.
4. Open **Vision controls**, select a video or connect a camera, choose enhancement/detection and press Play. Close the dialog to return to the detection map. Both displays share the same source and controls.
5. Use Record / Stop to save processed AVI clips, Snapshot + note for incidents, and Export logs for JSON. Results go to `ground_station/data/`; uploads stay under `ML/uploads/`.

Recording requires playing, fresh video. It stops automatically when the input pauses, powers off, disconnects, changes source, changes size or seeks; start a new recording after changing inputs. A paused AURA frame can still be saved with **Snapshot + note**, including its source and matching detection metadata. Notes distinguish video review from live camera evidence and simulated telemetry from live aircraft telemetry.

**Separate vision website:** choose **Vision website ↗** in the top bar to open the standalone AURA website in another tab (default `http://127.0.0.1:8766/`). The direct URL is also displayed in the Processed video card. **Vision controls** opens the embedded option. Both use the same pipeline: changing input, enhancement, exclusions or power in either place updates the shared output. The external link follows the configured `aura_port` automatically.

The launcher reuses AURA on port 8766 if already running. Otherwise it starts AURA using the existing `ML/.venv`, or `ML/.venv-portable` on a fresh machine. Closing the launcher stops only an AURA process that it created; an independently running dashboard remains running.

If a ground station is already running with different configuration, the launcher asks you to stop it or choose another port. Browser controls reconnect after a server restart; an aircraft control lease is never automatically restored. Use real JSON booleans (`true` / `false`) and numbers in configuration, without quotes around them.

If AURA was running before this integration was installed, stop and restart it once to load the new captured-frame timestamps and relative UI routes. An older service may remain viewable but is not accepted as fresh evidence for tracking/control.

## Default mode: video analysis with simulated telemetry

`config.integrated.json` selects the AURA video adapter and demo telemetry. The page explicitly marks this combination. GPS, attitude and vehicle movement are simulated; detections come from the selected real recording/camera. No aircraft connection is opened. Person injection is disabled because it would misrepresent real vision output; native demo mode remains available for synthetic scenario tests.

Power off stops enhancement/detection in AURA. It also invalidates video evidence in the ground station, which pauses an active simulated mission or blocks live movement. It does **not** switch off an aircraft or replace pilot controls. Paused/ended recordings do not provide fresh tracking evidence.

## Connect real telemetry later

Copy `config.live.example.json` to `config.json`, select the correct MAVLink endpoint and keep `live_control` and `live_missions` false for reception checks. Start with:

```powershell
.\ground_station\.venv\Scripts\python.exe .\ground_station\launch.py --config .\ground_station\config.json
```

Read [hardware notes](docs/HARDWARE.md) and [commissioning procedure](docs/COMMISSIONING.md), preserved from upstream. Pick the live capture device in Vision workspace. A file must never stand in for live visual feedback during aircraft control. This version preserves upstream ArduCopter GUIDED support; it does not claim compatibility with every flight controller/protocol. Network video URLs require a future capture adapter; they are not exposed by the current video selector.

Live movement still requires all upstream conditions: explicitly enabled configuration, valid lease, fresh telemetry/video, pilot-selected GUIDED, pilot arming and configured position/battery/flight limits. This integration does not arm, take off or issue commands during setup or testing. Aircraft location in incident logs is not object geolocation.

## Architecture and evidence

`ML/dashboard.py` owns capture → enhancement → detection and the GPU. The lightweight ground-station process consumes `/api/frame` over loopback, decodes the processed image, converts pixel boxes to normalized boxes and maps VisDrone `pedestrian` / `people` to the ground station's `person` category. The original label is retained in metadata. Existing filtered-out detections never reach person tracking.

Captured-frame timestamps and source epochs keep repeated UI revisions from counting as additional person sightings. Seeking or changing sources resets temporal evidence. Disconnection, pause, error and power-off remove usable evidence; only playing, fresh camera frames qualify as live input. These timestamps measure software read/inference age, not radio/camera-to-screen latency.

`/vision/` serves the existing AURA interface through the station. Its controls and streamed uploads use the station's same-origin/write-token protection. The proxy has fixed allowed routes and a configured loopback port; it cannot fetch arbitrary destinations. Uploads stream without buffering a whole file in RAM. The station records already annotated frames, so it does not draw duplicate boxes.

The map's optional online tile layers are unchanged. Local video processing and offline map mode need no cloud service. The station remains bound to loopback.

## Testing / troubleshooting

```powershell
cd ground_station
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m pip check
```

The suite includes upstream mission/control/API tests and integration contracts for label mapping, repeated frames, source changes, stale/paused/off inputs, recordings blocking live control, protected proxy operations, streaming upload and disconnect errors. No physical flight or USB device result is implied by passing software tests.

See [verification results and remaining limitations](VERIFICATION.md) for the completed software audit and the repeatable real-video integration check.

- **AURA disconnected:** start AURA; verify port 8766. The station reconnects automatically.
- **Port in use:** use `launch.py --port 8001`; change `aura_port` in a separate configuration if needed. The launcher does not terminate unrelated services.
- **Video present but no tracked candidates:** press Play and use detection; paused review is not live tracking. Slow CPU frames older than one second are rejected as tracking/control evidence.
- **Cannot start a mission:** inspect blockers; review the mode and source. A fresh real frame is required even for integrated simulation.
- **No extra GPU memory:** expected. Only AURA owns model inference; the station environment does not install Torch.

## Upstream attribution

Adapted from [rithinreddy0/aero-ground-station](https://github.com/rithinreddy0/aero-ground-station), commit `c2bd4cb8e5b321308b7b272c9ffb748e4f3a05b8`, cloned October 16, 2026. Core groundstation modules, web assets, tests and the two hardware documents were copied into this integration; see [original README](UPSTREAM_README.md). The source snapshot contains no LICENSE file; no new license is asserted for your friend's work. The original reference PDF/archive/images remain in the separate clone and are not duplicated here.
