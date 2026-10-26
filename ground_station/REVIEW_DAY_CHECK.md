# Project review check — 26 October 2026

**Result: recorded-video software demonstration is ready on this laptop. This does not certify perfect detections, target-GPS accuracy or aircraft operation.**

## Verified today

| Component | Result |
| --- | --- |
| Ground-station startup and AURA connection | Passed; localhost ports 8000 and 8766 |
| Existing GPU/model | RTX 2050, CUDA, FP16; original YOLOv8m VisDrone retained |
| Ground-station regression tests | 121 passed |
| ML/dashboard regression tests | 31 passed |
| Enhancement component tests | 39 passed |
| Eight enhancements × four detection settings | All 32 combinations passed with real inference |
| Play, pause, seek, EOF, source switching, power off/on | Passed running dashboard checks |
| Prediction cache and invalid-control rejection | Passed |
| Video upload | Short local test AVI accepted, decoded and listed through the station proxy |
| Processed recording and snapshot | AVI reopened successfully; JPEG incident created |
| Tracking | Real playback produced confirmed observations; recorded video correctly produced no GPS locations |
| Mapping logic | Geometry, uncertainty rejection, missing telemetry and duplicate/occlusion tests passed |
| Browser | Standalone and embedded vision work; sample-map filter works; closing embedded controls releases its connection; no console errors observed |
| Flight control | Disabled in the current demo configuration; no aircraft commands sent |

Total automated tests: **191 passed**. The separate 32-combination running-dashboard check and integration smoke check also passed. These counts measure software behavior, not detection accuracy.

## Corrections today

- Restored eight missing original videos from Git history with the user's approval.
- Fixed the manual-viewer test's relative sample path. It now finds its video when tests are launched from the project root. No inference code, weights or GPU packages were changed.

## Remaining limitations

- The retained detector still mistakes some trees and video-overlay text for objects. In the post-flight statistics screen, many overlay characters were labelled as cars. The exclusion regions only cover their configured areas; arbitrary overlay layouts are not automatically recognized. Show the region outlines and choose 0.25 or 0.50 according to the scene. A higher threshold can also miss real objects.
- Tracking IDs are temporary observations, not verified personal identity or a guaranteed unique-person count.
- Target mapping needs real time-aligned MAVLink pose, measured lens/mount calibration and surveyed ground. The preview map is synthetic. Actual geographic accuracy remains untested.
- Windows listed a USB2.0 HD UVC webcam and Bluetooth serial ports. No identifiable drone telemetry/capture setup was verified. Webcam capture, USB receiver reception, ELRS/MAVLink hardware and flight response were not exercised today.
- The station environment has no broken package requirements. The ML environment reports version conflicts for unused Google AI packages: google-ai-generativelanguage/grpcio-status versus protobuf, and google-genai versus websockets. All AURA checks passed despite these conflicts. These unrelated dependencies were left unchanged before the review.
- One upstream Starlette/httpx deprecation warning remains in the station tests.

## Demonstration sequence

1. Run **Start Ground Station.cmd** from the project root. Keep its service window running.
2. Open **http://127.0.0.1:8000/** for the detection workspace, or **http://127.0.0.1:8766/** for the standalone vision page.
3. The restored real_fpv_user_flight.mp4 recording is prepared paused around 1:05, with bilateral enhancement, 0.25 exclusion and region outlines. Press Play.
4. Compare the original view and enhancement choices; demonstrate exclusion off / 0.25 / 0.50. Explain that visibility improvement is not proof of improved detection accuracy.
5. Return to the detection workspace to show tracked observations, recording and Snapshot + note.
6. Use Preview sample locations only to explain how estimated coordinates and uncertainty circles will be displayed. Explicitly identify it as synthetic and exit the preview afterwards.
7. Explain that actual telemetry calibration and hardware trials are the next validation stage. Keep flight controls disabled.

The review-day check itself did not commit or push changes. The subsequent repository handover is documented in the root HANDOVER.md and Git history.
