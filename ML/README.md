# AURA detector milestone — 15 October 2026

## Main web dashboard: enhancement followed by detection

### Hardware portability

The dashboard detects the available CUDA device instead of depending on an RTX model name. Defaults remain GPU 0 with FP16 on this RTX 2050. `run_dashboard.ps1` accepts `-Device auto|cpu|cuda|cuda:N`, `-Precision auto|fp32`, and `-PythonPath` for a destination machine's environment. Auto falls back to clearly labelled CPU processing when CUDA is unavailable; explicit CUDA requests fail rather than silently switching. See [PORTABLE_SETUP.md](PORTABLE_SETUP.md) for required weights, a fresh environment and the official PyTorch installation selector. Do not copy this laptop's `.venv`, which references machine-specific paths. Existing packages and the running dashboard were left intact; updated hardware selection takes effect on the next launch.

### Start the page and use power

Open PowerShell and run:

```powershell
& 'C:\Users\azhar\CODES\MAJOR PROJECT\ML\run_dashboard.ps1' -Port 8766
```

Then open **http://127.0.0.1:8766/** in a browser. Keep PowerShell running. Opening `dashboard/index.html` directly will not work: the local Python service provides video input, enhancement and GPU detection. Without `-Port`, the launcher uses port 8765 instead. Ctrl+C in PowerShell stops the whole service.

The top-right **Power off** button stops frame processing, enhancement and detection after any in-flight frame completes, and releases an active camera. The webpage and local service stay available so **Power on** can resume using the selected source/settings. Recordings retain their position; live cameras reconnect. Model weights remain loaded in memory for quick restart, so Power off is not a GPU-memory unload or shutdown of the PC. The Play/Pause control remains separate from pipeline power.

**Exclusion control update:** the dashboard now offers FPV exclusion for every recording, upload and camera. Sessions start with Exclude 0.25; Original disables exclusion, and Detection off disables the detector. Changing source or connecting a camera preserves the selected mode. Regions and small-box limits scale proportionally from the original 640x480 profile to the input dimensions. This is geometric scaling, not automatic recognition of different overlay layouts; use the outlines to check alignment. The older standalone tools retain their strict profile-size checks. Verified with 11 tests and GPU API checks on a 320x240 uploaded recording, both exclusion thresholds, Original bypass and mode persistence across source changes. This supersedes source-specific restrictions described in earlier milestone notes below.

The main pipeline now uses the original Kaggle-trained VisDrone model. The alternative pretrained model experiments below remain separate.

```powershell
& 'C:\Users\azhar\CODES\MAJOR PROJECT\ML\run_dashboard.ps1'
```

Open **http://127.0.0.1:8765** and keep the launcher running. Ctrl+C stops the service. `-Port 8766` selects another port if necessary. No additional web framework packages are needed.

The webpage controls the existing eight enhancement methods, detection off/original/exclude 0.25/exclude 0.50, optional exclusion outlines, side-by-side original input, play/pause, seeking, restart and selection of recordings in the project root. It starts paused with raw enhancement and Exclude 0.25 detection. The source pixels pass through the chosen existing enhancer before the baseline detector receives them. Exclusion operates on predictions afterward. Changing settings while paused reprocesses that frame; settings do not require an OpenCV video window.

Zero-DCE uses the existing pretrained PyTorch FP16 implementation with four curves, explicitly on CUDA; this avoids the older viewer's working-directory-dependent ONNX selection. The seven other choices reuse the existing classes directly. Temporal restoration state resets on seek, source changes and enhancement changes; display-only changes reuse the enhanced frame. Errors are surfaced rather than silently substituting raw enhancement.

The dashboard offers telemetry exclusion for every input and scales the existing FPV regions to its dimensions. Original disables all exclusion filtering while keeping detections; Detection off bypasses detection entirely. The selected mode persists across source changes. Frames stay local. One shared backend session serves every open browser tab; changes in one tab affect the others. Browser previews poll at approximately 10 Hz and may skip intermediate processed frames; processing slows playback if it cannot keep up. Timings are per-frame processing observations, including cold-start costs, not camera-to-screen latency or sustained benchmarks.

The service binds only to 127.0.0.1. The older batch exporter and experiments remain available separately. Recording/export from the webpage, authentication for remote access, and flight control are not implemented.

### Input selection update

Use **Video from this PC** to choose MP4, MOV, AVI, MKV, WEBM or M4V recordings (up to 1 GB). The browser copies the selected file to a uniquely named directory under `ML/uploads`; the server validates that it can decode a frame before adding it to the recording list. Copies persist across restarts and never leave the PC. Codec support depends on OpenCV. Original files are never modified.

For a built-in webcam or USB webcam/capture adapter, choose **Camera 0–9** and click **Connect**. These are OpenCV device indexes, not an enumeration of installed devices. Windows uses DirectShow. The service requests 640x480, but a device may negotiate a different resolution. Windows camera/privacy permissions and exclusive use by another app can prevent connection; the dashboard displays the error. Pause releases the device, Resume opens it again, and Disconnect returns to the supplied recording. Live cameras cannot seek/restart. Exclusion is available for cameras too, with regions scaled to the delivered frame size and the selected mode preserved when connecting.

`video_sources.py` separates camera/file opening from enhancement and detection so future adapters can be added. Network/RTSP streams and proprietary receiver SDKs are not implemented. A USB receiver must expose a standard camera/capture interface to use the current camera option. Physical hardware operation has not been verified here; camera tests use simulated devices. Video upload/decode, source selection, invalid-file rejection, and the browser file picker were tested successfully. Eight source/mode/filter tests passed.

`check_dashboard.py` checks a running dashboard: all 32 enhancement/detection combinations, pause/play, seek, end-of-file, source changes and invalid controls. Results are saved to `outputs/dashboard_checks.json`. Run enhancement regressions from the `feeed enhancement` folder because the older viewer tests use relative sample paths.

Verified in this milestone: 32 API combinations and playback/source checks passed; all 39 enhancement regression tests passed from their expected working directory; four existing mode/filter tests passed. Browser controls were exercised for enhancement selection, original comparison, exclusion switching and play/pause. A rendered preview is saved at `outputs/dashboard_preview.png`. No existing enhancement algorithm was changed for this integration.

## Interactive model experiments

The original VisDrone setup is preserved as **Model Experiment 1**. Launch the new interactive recording viewer from the project root:

```powershell
.\ML\run_experiment_1.ps1
```

Click the three buttons or press **1 / 2 / 3** while playing or paused:

| Key | Mode | Behavior |
|---|---|---|
| 1 | Original 0.25 | Original detector output, no telemetry exclusion |
| 2 | Exclude 0.25 | Telemetry exclusion, confidence at least 0.25 |
| 3 | Exclude 0.50 | Telemetry exclusion, confidence at least 0.50 |

Space pauses/resumes, R restarts, Q/Escape exits. Start in a chosen mode with `-Mode 2`. Inference always runs at 0.25, so switching back restores candidates immediately without reloading weights. Buttons occupy a separate strip above the feed. Amber regions show possible blind spots. The profile is specific to this recording's 640x480 layout; other feeds need calibration even if their dimensions match. This is recording playback with inference, not live receiver integration. Slow inference slows playback rather than dropping frames.

Each session saves model/source/profile metadata, original predictions, display decisions and mode changes under `outputs/interactive_*`. It does not save a new video; the older `run_detection.ps1` remains the batch video exporter. All eight existing enhancement modes remain unchanged and separate.

Four mode/filter tests passed and all three modes rendered successfully in a GPU smoke run. PowerShell launcher parsing and experiment lookup were checked. Native mouse/keyboard interaction has not been manually exercised in this session.

## Pretrained model screening

Experiments are registered in `experiments.json`. Try any downloaded candidate with the same interactive controls:

```powershell
.\ML\run_model_experiment.ps1 -Experiment 2
# Experiment 3 = YOLO26n; Experiment 4 = YOLO26s
```

Official pretrained weights were downloaded and tested on exactly the same 211 frames: every fifth frame in 0–749 and 1750–2049, plus frame 543. All runs used raw images, input size 640, confidence 0.25, FP16, five warmups and synchronized CUDA timing. The four-panel comparison applies the same OSD exclusion to each model; each JSONL also retains its unfiltered predictions.

| Experiment | Model / training | Median prediction | P95 prediction |
|---|---|---:|---:|
| 1 | YOLOv8m / user VisDrone | 19.02 ms | 20.41 ms |
| 2 | YOLO11s / official COCO | 11.85 ms | 14.99 ms |
| 3 | YOLO26n / official COCO | 13.78 ms | 37.30 ms |
| 4 | YOLO26s / official COCO | 16.39 ms | 31.33 ms |

Review `outputs/pretrained_screening_01/comparison.mp4`, its snapshots, and `report.json`. All 211 video frames and four prediction logs were verified. The video is sampled playback at source FPS/5 with a segment jump, not the complete flight. These are single sequential screening runs, not sustained latency guarantees; thermal/background activity can affect them.

At frame 2000, YOLO11s detects both visible people (0.48 and 0.35), YOLO26s detects one (0.42), and Experiment 1 and YOLO26n miss them at 0.25. At frame 543, all four detect a person, but the COCO models can produce overlapping person boxes. COCO candidates also mistake telemetry for kites. These specific examples favor further investigation of YOLO11s, not a claim of best overall accuracy. The 0.50 mode would suppress both YOLO11s person detections at frame 2000.

There are no FPV ground-truth annotations yet, so no precision/recall/mAP or overall winner is reported. COCO's 80 classes and VisDrone's 10 differ; comparing total box counts is not an accuracy comparison. No model was retrained and Experiment 1 was not replaced.

### Candidate selection and customization

- [YOLO11](https://docs.ultralytics.com/models/yolo11/) provides pretrained sizes and custom training/export support. YOLO11s is the first candidate for a labelled FPV evaluation based on these local examples and measured speed.
- [YOLO26](https://docs.ultralytics.com/models/yolo26/) offers nano/small pretrained variants and custom fine-tuning. Both were tested locally. Published benchmark advantages do not guarantee FPV quality or laptop latency. Its P2/P6 architectures do not have released pretrained weights, so those are training experiments rather than ready-to-test pretrained replacements.
- [RT-DETR](https://docs.ultralytics.com/models/rtdetr/) is a useful transformer comparison for a later accuracy-oriented trial. It has not been downloaded or benchmarked here; measure memory and latency before assuming it fits the live budget.

Next customization should keep each pretrained baseline intact, label representative FPV clips (people plus empty scenes/telemetry/interference), split by flight to avoid near-duplicate leakage, and compare fine-tuned models on held-out flights. Confidence calibration and optional person-only display can follow that evaluation. Corrected CBAM training remains a separate, untrained experiment below.

The existing Kaggle-trained VisDrone detector now runs on the RTX 2050 and produces annotated video and per-frame detection records. Visual review found false detections on FPV telemetry/interference and missed people. This is a working baseline, not a validated rescue detector.

## Review the results

- Full recording: `outputs/visdrone_raw_640_full/annotated.mp4`
- Measured report: `outputs/visdrone_raw_640_full/report.json`
- Individual detections: `outputs/visdrone_raw_640_full/detections.jsonl`
- Higher-resolution comparison: `outputs/visdrone_raw_960_people_segment/annotated.mp4`
- Review frames are saved alongside each video.

| Run | Frames | Median prediction | P95 prediction | Offline processing FPS |
|---|---:|---:|---:|---:|
| 640 inference, full recording | 4,496 | 23.00 ms | 30.07 ms | 34.37 |
| 960 inference, frames 1750–2049 | 300 | 44.78 ms | 49.10 ms | 20.15 |

The two timing runs cover different frame ranges. The 960 run is a targeted diagnostic, not a controlled full-dataset accuracy comparison. At frame 2000, both settings missed visible people. Increasing input size did not resolve that example.

The full output was decoded again: all 4,496 frames were readable and matched 4,496 JSONL records. Source FPS is approximately 24.97; output playback preserves that rate. Prediction timing includes preprocessing, model execution and postprocessing, with CUDA synchronization and five excluded warmup predictions. Offline FPS also includes decoding, drawing and saving. Neither measures live receiver-to-screen latency. No FPV accuracy score is claimed without annotations. Detection counts repeat objects across frames.

## Run locally

From the project root in PowerShell:

```powershell
.\ML\run_detection.ps1
# Optional live preview while processing a recording:
.\ML\run_detection.ps1 -Show
# A different recording:
.\ML\run_detection.ps1 -Source 'C:\path\flight.mp4'
```

Each launch creates a new output directory. Press Q or Escape to stop an enabled preview. The standalone detector currently accepts recordings, not live receiver input. It runs on raw frames so enhancement can be evaluated separately later.

For custom settings:

```powershell
& .\ML\.venv\Scripts\python.exe .\ML\detect_video.py --imgsz 640 --conf 0.25 --max-frames 300 --output .\ML\outputs\my_review
```

CUDA is required by default. The program stops if CUDA is unavailable; CPU requires explicit `--device cpu`. Class mapping is checked before processing. Existing result directories are protected against accidental overwrite.

## Environment

Verified: Python 3.11.9, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, Ultralytics 8.4.127, OpenCV 4.13.0, RTX 2050 4 GB, NVIDIA driver 617.14.

`ML/.venv` is an isolated overlay environment that reuses installed packages. Its `aura_existing_gpu.pth` points to `C:\Users\azhar\CODES\ai\.venv\Lib\site-packages` for CUDA PyTorch. It also inherits system packages. It is therefore not portable or fully self-contained. Removing that external environment will break this setup. For another computer, create a fresh environment, install a compatible CUDA PyTorch/torchvision pair, then install `ML/requirements.txt`. No existing GPU package was replaced.

## Checkpoint provenance

`models/yolov8m_visdrone_recovered.pt` was reconstructed from the files in `models/cbam_best/best`. Each archive member was verified byte-for-byte against its original, and the recovered model loaded and inferred successfully.

SHA-256: `200f73fd846db1ceb46a55614dc90081b4ecfa1393f3f4cfd6152dd93a5928c9`

The checkpoint is YOLOv8m with 10 VisDrone classes and no CBAM modules. Its saved training history covers 50 epochs. Saved metrics are precision 0.59927, recall 0.48873, mAP50 0.50205 and mAP50–95 0.30890; these have not been independently reproduced locally.

Notebook reviewed: https://www.kaggle.com/code/azhar77777/notebook5c355ff2e2

The notebook manually wrapped layers before `model.train()`. Ultralytics reconstructed the network in `trainer.get_model(...)`, discarding these additions. The training log lists standard YOLOv8m layers. Its YOLO26n download was an AMP check. The missing F1 image error occurred after training and checkpoint export.

## Corrected CBAM training

`aura_cbam.py` inserts CBAM inside the trainer's model-construction step, after pretrained weights have loaded. It preserves YOLO layer-routing metadata and obtains the actual output channel counts. A startup callback checks that all three CBAM blocks are present and all attention parameters are in the optimizer. Export reloads the best checkpoint and checks the blocks again before giving it a verified filename.

`test_cbam.py` passed checks for trainer construction, preservation of baseline weights, optimizer inclusion, nonzero finite gradients through each attention block, and checkpoint save/reload/inference. These are software smoke checks; a complete CBAM training run has not been performed.

To use on Kaggle, upload `aura_cbam.py` and `train_cbam.py` together and keep their directory importable. Install `ultralytics==8.4.127`, supply an existing YOLOv8m checkpoint and a dataset YAML whose paths are correct, then explicitly run:

```text
python train_cbam.py --data /actual/path/VisDrone.yaml --weights /actual/path/yolov8m.pt --output /kaggle/working/aura_cbam_new
```

This starts a new 50-epoch experiment at 1024 with batch 6. The initial checkpoint choice affects the comparison: starting from the AURA VisDrone baseline is additional fine-tuning; starting from COCO weights follows the earlier initialization. Keep `aura_cbam.py` with the resulting checkpoint for loading. The implementation preserves the notebook's average-pooling channel-attention variant; do not describe it as an independently validated improvement or an exact reproduction of canonical CBAM. Resume/multi-GPU training has not been validated.

## Enhancement fixes and checks

- All eight viewer modes remain selectable.
- Reduced-curve Zero-DCE now loads the saved pretrained weights using a project-relative absolute path. It selects the corresponding pretrained output channels for fewer curve iterations.
- Missing neural checkpoints fail explicitly instead of leaving random weights active.
- The existing analytical fallback, when PyTorch is absent, is identified in returned metadata.
- Viewer help now lists the correct eight modes.
- Existing enhancement tests plus two new regression tests: **39 passed**.
- Pre-existing changes to restoration v2 and scenario evaluation were preserved.

## Next review decision

Inspect the output video before integrating this model into the live pipeline. The next quality milestone should use a small annotated FPV review set, explicitly including telemetry and interference examples. That will allow meaningful comparison of the baseline, another pretrained detector, and a correctly trained CBAM model. Automatic OSD suppression must be configurable because masking image regions can also hide real targets. Dashboard, live receiver integration, tracking and supervised flight assistance remain future milestones.

## FPV telemetry review update

`outputs/osd_review/comparison.mp4` compares the same cached predictions in three panels: original confidence 0.25, optional telemetry exclusion at 0.25, and telemetry exclusion at 0.50. The video covers source frames 0–749 followed by 1750–2049; the source frame number is displayed. This is a postprocessing comparison, not retraining or a new speed benchmark.

The optional `profiles/user_flight_osd.json` is calibrated for this 640x480 recording. It excludes small boxes mostly inside specified timer, battery, link-quality, reticle and moving-horizon regions. The original image and model input are unchanged. Amber outlines show the configured regions. Every excluded box is retained in the JSONL logs with its reason. Boxes that extend well outside an overlay region or exceed its configured glyph size remain eligible.

Across all 4,496 cached frames, the profile excluded 12,675 of 37,784 predicted boxes. These are filtering counts, not confirmed false positives. At confidence 0.50, 8,584 boxes remain, including 849 human-labelled boxes, versus 4,596 human-labelled boxes with OSD filtering at 0.25. This large reduction is why 0.50 is an explicit comparison choice, not a claimed accuracy improvement or new default. The visible pedestrian at source frame 543 (confidence 0.5098) survives both filters.

Limitations: the moving-horizon region is a geometric approximation; small real targets in it can also be excluded. Different camera layouts and post-flight STATS screens require separate calibration. Tree/background mistakes and missed people remain unresolved model errors. The filter rejects different image dimensions and is disabled unless explicitly selected.

```powershell
.\ML\run_detection.ps1 -Show -OsdProfile .\ML\profiles\user_flight_osd.json
# Optional stricter confidence comparison:
.\ML\run_detection.ps1 -Show -OsdProfile .\ML\profiles\user_flight_osd.json -Confidence 0.50
```

Verification: three filter tests pass (including the documented small-target blind spot), live detector integration succeeded on three frames, the comparison decodes to 1,050 frames, and the review log covers all 4,496 source frames.
