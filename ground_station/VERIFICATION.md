# Integration verification — October 19, 2026

- 121 ground-station tests passed, including mission/control/API tests, AURA adapter/proxy contracts, calibrated geometry, duplicate suppression, ambiguous crossings, short occlusions, track expiry, telemetry epochs and mapping API regressions.
- All 31 AURA tests passed, including buffered-camera timestamps and CBAM training/checkpoint preservation. Total: **152 passing tests**.
- Browser scripts passed syntax checks. The test runner reports one upstream Starlette/httpx deprecation warning; no test failed.
- The independent ground-station environment passed `pip check`. Its installed versions are recorded in `requirements-lock.txt`; AURA's CUDA environment was not modified.
- A real running integration check passed on the RTX 2050 using the existing VisDrone weights: raw, bilateral and Zero-DCE processing; exclusion changes; readable processed AVI recording; incident JPEG; power-off and pause removing usable tracking evidence.
- Browser verification showed the embedded AURA controls changing enhancement/playback and the processed feed appearing in the new detection workspace. Real video is labelled separately from simulated telemetry.
- Browser verification also covered automatic reconnection after a server restart, disabled controls while disconnected, refreshed embedded write tokens, and saving a note with its paused review frame. Both standalone and in-page vision access remain available.
- The upstream clone remains unchanged. This verification preceded publication; see the root HANDOVER.md and repository history for the later handover update.

## Detection workspace verification

- The main page focuses on observations, a location map, processed video and small operator-approved movements. The previous mission workspace remains at `/advanced`.
- Known synthetic geometry tests recover a target 20 metres north/east of the aircraft, verify downward-camera and roll behavior, and reject clipped boxes, near-horizon rays, missing pose, incorrect aspect ratio, excessive error and targets outside the surveyed area.
- Actual encoded MAVLink messages were decoded into independent position/attitude/GPS histories. Tests cover alignment with configured delays, stale samples and missing attitude. This is a software check, not measured radio timing or field accuracy.
- Recorded footage produced confirmed session tracks in the running GPU pipeline. All remained unlocated, and the GeoJSON export stayed empty. Pausing showed last-seen observations; browser Dismiss / Show dismissed / Restore worked.
- Incident snapshots only include object coordinates when both their source session and frame match the captured image. Regression tests reject older detection frames and different source sessions.
- Synthetic sample locations, category filtering and uncertainty circles were checked in the browser. Sample data never enters live observations, exports or calibration. Movement controls remained disabled with hardware pending and in sample mode.
- Restarting the local service reconnected the page. Video stream startup no longer opens a duplicate stream, disconnects clear its old URL, and closing Vision controls unloads the hidden polling page while the shared pipeline continues running. Browser scripts passed syntax checks and the final page had no console errors.
- The hardware guide explains fixed analog capture, the separate ELRS/MAVLink path, measured lens/mount/ground calibration and controller compatibility. Current recordings, enhancement, exclusion and paused position were restored after testing.

These checks do not establish unique-person identity, detector accuracy or exact target GPS. The fixed-camera estimator uses a surveyed flat ground plane; its displayed error is a sensitivity estimate under supplied assumptions, not a guaranteed confidence bound. Actual USB reception, flight-controller compatibility, variable link latency and estimates against surveyed physical targets remain to be tested with the hardware.

## Issues corrected in the audit

- Recording stops cleanly when video pauses, powers off, disconnects, changes dimensions, changes source or seeks. One file cannot silently mix different input sessions. Frame access and metadata capture use the same lock.
- Incident notes can include the displayed paused AURA frame, matching detections, source/frame identity and separate video/telemetry context. Native detections from an older frame are omitted instead of being attached to the wrong image.
- Buffered camera frames retain the timestamp from capture, so waiting in a buffer does not make old footage appear fresh. Native recorded files are paced at their own frame rate and stop at EOF.
- Person confirmation requires consecutive detection frames. Missing or disabled detections clear tracks; repeated frames cannot increase confirmation counts or keep a confirmation alive.
- Waypoint hold time resets when the aircraft leaves the target tolerance. Running missions cannot restart accidentally, revised routes cannot resume an older plan, and mission completion/cancellation no longer leaves an outdated movement status.
- Configuration rejects ambiguous booleans, missing explicitly selected files, unknown fields, invalid OSD rectangles and non-finite numbers. Invalid telemetry cannot pass live-control checks. Native recorded files cannot stand in for live camera feedback.
- Dashboard sessions recover after a server restart. Writes refresh their token, the video connection reconnects, and controls recover when the service returns. The launcher reports an existing server with incompatible configuration instead of silently reusing it.

To repeat the real-video check, launch **Start Ground Station.cmd**, leave simulated telemetry selected and stop any mission/recording, then run from the project root:

```powershell
.\ground_station\.venv\Scripts\python.exe .\ground_station\check_integration.py
```

The check creates a short clip and incident in `ground_station/data/` and restores the selected pipeline controls. It uses recorded footage, not an aircraft connection.

Physical USB capture, MAVLink radio reception, aircraft response and real flight have not been tested here. Other laptops have not been benchmarked. Network-stream input still needs its capture adapter. The original VisDrone model is retained; trees, overlays and unfamiliar scenes can still produce false detections, and no model-accuracy improvement is claimed. Live control remains disabled in the supplied configurations.
