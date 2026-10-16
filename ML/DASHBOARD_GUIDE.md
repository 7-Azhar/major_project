# AURA dashboard

## Start and stop

Double-click **Start AURA.cmd** in the project folder. It opens **http://127.0.0.1:8766/** and uses the existing `.venv`, or `.venv-portable` if the first environment is absent. Keep its terminal window running. A second launch opens the existing AURA service on that port without loading another detector.

Alternatively, from the project folder:

```powershell
.\ML\run_dashboard.ps1 -Port 8766 -OpenBrowser
```

Opening the HTML file directly cannot run the pipeline: the Python service supplies frames and processing. Ctrl+C in the launcher stops that service. **Power off** in the webpage stops processing after the current frame finishes, releases an active camera when its driver returns, and leaves model weights loaded for a quick restart. It does not shut down the PC. **Power on** resumes the selected settings. A recording retains its position through power-off; a camera reconnects.

If the service restarts, the page reconnects automatically and shows the new session's settings. A fresh server starts paused with the first available recording, raw enhancement, and Exclude 0.25. The selected file and frame position are not saved across server shutdown. Every browser tab controls the same local pipeline.

## Choose an input

- **Saved recording / active camera:** choose a known local file, or choose the empty input to stop viewing it.
- **Video from this PC:** select MP4, MOV, AVI, MKV, WEBM or M4V, up to 1 GB. A validated copy stays under `ML/uploads` across restarts. Original files are unchanged. The upload can be cancelled; nothing is sent to a cloud service.
- **Webcam / USB capture device:** choose Camera 0–9 and Connect. These are device numbers, not a scan of installed cameras. A USB FPV receiver must expose a standard webcam/capture interface. Windows uses DirectShow and requests 640×480; the device may return a different resolution.

Pause releases a camera; Resume opens it again. Disconnect returns to your most recent recording, or an empty input if none is available. Seeking and Restart apply only to recordings. Camera capture retains only the latest frame, so a slower processing stage drops old frames instead of accumulating a queue. A driver that blocks inside a read can delay camera release; the service prevents duplicate readers for that device until it closes. If connection fails, check Windows camera permissions, the cable, and other apps using the device.

## Enhancement and detection

The original Kaggle-trained **YOLOv8m VisDrone** detector is still the main model. All eight existing enhancement methods remain available. Enhancement runs before detection. Use **Compare with original feed** to inspect the effect on the same frame.

| Detection choice | Result |
| --- | --- |
| Detection off | Enhancement only |
| Original · exclusion off (0.25) | Predictions with confidence at least 0.25; no telemetry filter |
| Exclude telemetry · 0.25 | Apply the FPV telemetry filter to predictions at 0.25 |
| Exclude telemetry · 0.50 | Apply the same filter and require confidence at least 0.50 |

Exclusion is available on every video and camera, and stays selected when the input changes. Amber outlines show the configured regions scaled from the original 640×480 layout. This does not recognize or calibrate a new telemetry layout. Check their alignment; use Original to disable exclusion completely. Small real objects inside those regions can be hidden, and 0.50 can remove real low-confidence detections.

On a paused frame, changing thresholds or outlines reuses the same predictions. A new frame or enhancement method triggers fresh processing. Seeking, changing source, or switching enhancement resets temporal enhancement history. An unchanged paused frame is not repeatedly transmitted to the browser.

The timings report the cost of processing that frame, including cold starts. When predictions are reused, the page says so and retains their original processing time. These values are not camera-to-screen latency, sustained FPS, or detection-accuracy measurements.

## Review and navigation

- **Save review frame** downloads one self-contained HTML file with the displayed original and output JPEGs, frame number, source, selected settings, hardware, visible detections, and excluded detections with reasons. Open it later in a browser without AURA running. It is a snapshot, not a video recording or a labelled accuracy evaluation.
- **Full screen** enlarges the monitor when supported by the browser. Use **Exit full screen** or Escape to return.
- **Space** plays/pauses when focus is outside the controls. **Replay** starts a finished recording again.
- On narrow screens the monitor appears first; **Input & settings** jumps to the controls below. Each control section can collapse.

For other GPUs or CPU use, see [PORTABLE_SETUP.md](PORTABLE_SETUP.md). This machine retains its existing RTX 2050 environment and default GPU/FP16 behavior. No model weights were retrained or replaced by this dashboard update.

## Verification for this update

- 22 regression tests passed for hardware selection, filtering, source validation, stale-result rejection, same-frame caching, and simulated latest-frame camera capture.
- All 32 enhancement/detection combinations passed against the RTX 2050 service, with playback, seeking, end-of-file, source changes, power-off/resume, and unchanged-preview checks.
- CPU/FP32 smoke runs passed for raw and Zero-DCE enhancement with detection.
- Browser checks passed for comparison, threshold and outline controls, power-off/on, local upload, saved review contents, and automatic reconnection after restarting the server. The 390-pixel-wide layout has no horizontal overflow and puts the monitor first.
- Invalid uploads and foreign-origin control requests were rejected. A repeated launch reused the running server without constructing another model.

Physical USB-camera operation and RTX 4050 hardware have not been tested here. The original detector's false positives and missed objects remain model limitations; these changes improve operation and review, not measured accuracy.

To repeat the API checks on a **test** server (the checks change its source and playback settings):

```powershell
& .\ML\.venv\Scripts\python.exe .\ML\dashboard.py --port 8767
# In another terminal:
& .\ML\.venv\Scripts\python.exe .\ML\check_dashboard.py --port 8767
```
