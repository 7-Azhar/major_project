# AURA project handover

Prepared 26 October 2026 for continued development on another laptop.

## Get running

Clone https://github.com/7-Azhar/major_project.git or download and fully extract its ZIP. Install 64-bit Python 3.11 with the Windows Python launcher. Run **Setup AURA.cmd**, then **Setup Ground Station.cmd**. Setup needs internet and several GB of space. Do not copy the original laptop's virtual environments.

- **Run Both Workspaces.bat** starts the shared vision pipeline and detection workspace, then opens both pages: http://127.0.0.1:8000/ and http://127.0.0.1:8766/.
- **Run Vision Dashboard.bat** starts only vision and opens http://127.0.0.1:8766/. It requires only Setup AURA.cmd.
- Keep the server window open. Ctrl+C stops the service in that window. Reopening a launcher reuses matching existing services; if it exits after reuse, keep the original server window open.
- The combined launcher stops the vision child it created when it shuts down. An independently started vision service is left running.
- A fresh installation uses ML/.venv-portable. Existing ML/.venv is preferred on the development laptop. NVIDIA or CPU selection is automatic; see ML/PORTABLE_SETUP.md for explicit options. Performance depends on the hardware.

Both interfaces control the same source, enhancement and detector. A change in one appears in the other. Do not run a second camera reader for the same receiver.

## Current implementation

Video/file/USB capture → selectable enhancement → original YOLOv8m VisDrone detection → optional overlay exclusion → temporary object tracks → review, recording and estimated-location map.

The main detector is `ML/models/yolov8m_visdrone_recovered.pt`. Keep this working baseline. The CBAM and other model experiments remain separate: do not relabel the baseline as CBAM or claim an improvement without evaluation. Eight enhancement methods and four detection modes are available. Enhancement changes appearance; a detection-accuracy benefit has not been demonstrated by these software tests.

| Location | Purpose |
| --- | --- |
| ML/dashboard.py, ML/dashboard/ | Capture service and standalone web interface |
| ML/frame_processor.py, detection_modes.py, osd_filter.py | Enhancement/detection execution, caching and exclusion |
| ML/hardware.py, setup_aura.py, ML/verify_setup.py | Hardware selection, portable installation and asset/inference checks |
| feeed enhancement/app/ | Enhancement algorithms and neural enhancer |
| ground_station/groundstation/aura.py | Consumes processed AURA frames; no second GPU model |
| ground_station/groundstation/observations.py | Duplicate suppression, temporal confirmation and track review |
| ground_station/groundstation/geolocation.py | Fixed-camera, surveyed-flat-ground location estimator |
| ground_station/groundstation/telemetry.py | MAVLink reception and time-aligned pose history |
| ground_station/groundstation/station.py, control.py | Service orchestration and guarded operator movement |
| ground_station/web/overview.* | Main detection workspace |
| ground_station/web/index.html, app.js | Previous mission workspace at /advanced |
| ground_station/docs/TARGET_MAPPING.md | Calibration, limitations and hardware integration |

The integrated default is **demo telemetry with actual selected video**. Simulated GPS is withheld from the main map. The sample-map button is a labelled illustration and never supplies real observation locations. The map exports only confirmed, located, non-dismissed observations.

## Hardware information still needed

Known: fixed analog camera, a planned compatible laptop video receiver/capture device, and ExpressLRS (ELRS). Exact flight controller, firmware, radio hardware and capture device are still unconfirmed.

1. Confirm those models and firmware versions. Check that the video receiver exposes a usable camera feed to Windows/OpenCV and that the radio arrangement actually transports MAVLink.
2. Copy `ground_station/config.live.example.json` to the ignored `ground_station/config.json`. Configure the endpoint; keep live_control and live_missions false. Start with `ground_station/.venv/Scripts/python.exe ground_station/launch.py --config ground_station/config.json --open-both`.
3. Validate real video and telemetry reception first. Camera index numbers are selectable inputs, not a verified list of devices.
4. Measure lens calibration, fixed mounting angles, capture/telemetry delays and ground elevation/extent. Load the completed JSON through Hardware & mapping setup. Never mark example/default values verified.
5. Check estimates against surveyed physical targets. Aircraft GPS is not a detected person's GPS. Current estimates assume a visible ground-contact point and flat surveyed terrain; uncertainty circles are not guaranteed error bounds.
6. Confirm supported flight modes/controller behavior and bench-test before considering movement. The inherited control path expects ArduCopter GUIDED behavior; compatibility with unknown firmware is not established. Existing pilot/telemetry/video/lease gates remain required.

Do not enable autonomous person following or present operator-approved requests as confirmed aircraft movements. No physical flight validation has been completed here.

## Known limitations and next work

- Trees and overlay characters still produce false detections. Exclusion uses configured geometric regions, not automatic recognition of every HUD layout. Different layouts need their own measured regions; exclusion can hide real targets.
- IDs represent temporary observations, not personal identity. Crossings, camera motion and occlusion can split or swap tracks; a globally unique-person count is not guaranteed.
- Recordings are never matched to current aircraft GPS. Offline geolocation needs a future synchronized recorded-telemetry workflow.
- Network-stream inputs need a future adapter. Do not claim arbitrary stream URLs already work.
- Preserve the working baseline; collect labelled representative footage and evaluate precision/recall before choosing a replacement detector or asserting enhancement gains.
- Test a fresh install on the friend's actual laptop and perform hardware calibration/reception tests before flight work.

## Included and excluded

Model assets, original sample videos, application code, tests, setup scripts and documentation are in the repository. No Git submodules or LFS installation is required. The copied integrated ground station is self-contained; the separate `external/aero-ground-station` reference clone is not needed to run it.

Ignored local material includes virtual environments, uploads, recordings, session databases, measured calibration, live config, local datasets and environment files. These are created locally or supplied deliberately when needed; they are not missing application dependencies. Never commit credentials or operational location logs.

Upstream ground-station attribution and original README are preserved in `ground_station/README.md` and `ground_station/UPSTREAM_README.md`. No new license is asserted for the friend's upstream work.

## Verification

On 26 October: the review check passed 121 station tests, 31 ML tests and 39 enhancement tests (191 total), plus all 32 enhancement/detection combinations and real GPU recording/snapshot/power checks. The handover added seven launcher tests: the final station and ML suites passed 125 and 34 tests respectively, bringing the total to 198 with the unchanged enhancement suite. All eight manifest assets passed SHA-256 checks. Both batch launchers were exercised from another working directory. See `ground_station/REVIEW_DAY_CHECK.md` for scope and limitations.

Use the AURA environment created on this laptop (replace .venv-portable with .venv if appropriate):

```powershell
.\ML\.venv-portable\Scripts\python.exe -m pytest ML -q
.\ML\.venv-portable\Scripts\python.exe -m pytest 'feeed enhancement/tests' -q
.\ground_station\.venv\Scripts\python.exe -m pytest ground_station/tests -q
.\ML\.venv-portable\Scripts\python.exe ML/verify_setup.py --assets-only
```

For running-service checks, first launch both workspaces in demo mode with no active mission or recording:

```powershell
.\ML\.venv-portable\Scripts\python.exe ML/check_dashboard.py --port 8766
.\ground_station\.venv\Scripts\python.exe ground_station/check_integration.py
```

These checks change demo playback/settings and create test outputs. Do not run them during an aircraft session. The dashboard check leaves its own demonstration settings; the integration check restores settings but advances the recording position.

The original ML environment has conflicts among unrelated Google AI packages. The portable setup installs the project's own declared dependencies; do not copy or replicate the old environment wholesale. No field-accuracy or cross-laptop performance guarantee follows from passing software tests.
