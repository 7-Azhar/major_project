# Advanced setup on another computer

For the simple Windows installation, follow the [root README](../README.md) and run **Setup AURA.cmd**, then **Start AURA.cmd**.

## What a clone includes

The repository includes the dashboard, full enhancement source, baseline and experiment detector weights, Zero-DCE weights/ONNX exports, sample recordings, tests, and selected reference reports. No submodule or Git LFS step is needed. `assets-manifest.json` records the required weights/demo hashes. Training datasets, local Python environments and generated output archives are intentionally not part of the clone.

Do not copy the original laptop's `.venv`: it references an external installation. `setup_aura.py` always builds `ML/.venv-portable` using Python 3.11 and does not alter `ML/.venv`.

## Explicit installation

```powershell
py -3.11 setup_aura.py --device cuda
# Or CPU-only:
py -3.11 setup_aura.py --device cpu
# Or the supplied CUDA 12.8 package pair:
py -3.11 setup_aura.py --device cuda --cuda-channel cu128
```

The default NVIDIA pair is torch 2.5.1 / torchvision 0.20.1 from the official CUDA 12.1 index. The CUDA 12.8 option installs torch 2.7.1 / torchvision 0.22.1. Automatic mode selects CUDA 12.8 for detected RTX 50-series devices; otherwise it selects CUDA 12.1 when `nvidia-smi` finds an NVIDIA device, or CPU when it does not. These choices are pinned, not an attempt to install whichever version is newest.

See the [official PyTorch version commands](https://docs.pytorch.org/get-started/previous-versions/) and [installation selector](https://pytorch.org/get-started/locally/) if a different GPU/driver requires another runtime. Install into the portable environment using its Python executable, never by replacing packages in the original developer environment:

```powershell
.\ML\.venv-portable\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\ML\.venv-portable\Scripts\python.exe -m pip install -r ML/requirements.txt
.\ML\.venv-portable\Scripts\python.exe -m pip check
.\ML\.venv-portable\Scripts\python.exe ML/verify_setup.py --device cuda
```

A successful package installation is not sufficient: `verify_setup.py` checks actual device execution, reads the demo recording, loads the baseline model, and runs raw and Zero-DCE enhancement followed by detection. A compatibility or missing-asset error stops setup with a message. It does not silently replace trained weights.

## Select hardware at launch

```powershell
.\ML\run_dashboard.ps1 -Port 8766 -PythonPath .\ML\.venv-portable\Scripts\python.exe -OpenBrowser
# GPU 1 or full precision:
.\ML\run_dashboard.ps1 -Port 8766 -PythonPath .\ML\.venv-portable\Scripts\python.exe -Device cuda:1 -Precision fp32
# Explicit CPU:
.\ML\run_dashboard.ps1 -Port 8766 -PythonPath .\ML\.venv-portable\Scripts\python.exe -Device cpu
```

If PowerShell script execution is restricted, run the equivalent directly:

```powershell
.\ML\.venv-portable\Scripts\python.exe ML/dashboard.py --port 8766 --device auto --open-browser
```

Dashboard `auto` selects GPU 0 when CUDA is available, otherwise labelled CPU processing. Explicit CUDA requests fail if unavailable. FP16 is used on compatible CUDA hardware unless FP32 is requested. The RTX 2050 defaults remain GPU 0, FP16 and image size 640; another GPU's speed/memory characteristics may differ. The dashboard reports the actual device and precision.

Each laptop hosts its own service on 127.0.0.1. Copying a URL to another laptop does not connect it to the original machine. Keep the launcher running; Ctrl+C stops it. A second Start AURA launch on the same port opens the existing AURA service without allocating another model.

## Experiment and legacy tools

The experiment viewer also supports `--device auto|cpu|cuda|cuda:N` and `--precision auto|fp32`; PowerShell experiment launchers use the available local/portable environment. The raw batch detector accepts explicit `cuda` or `cpu`. The historical enhancement viewer has optional ONNX dependencies and separate provider requirements; they are not needed for the main dashboard. DLL discovery now uses the selected Python environment and `CUDA_PATH`, rather than a developer-specific user folder.

RTX 4050 and other destination hardware still need the included execution check. Multiple-GPU selection and the capture buffering logic have simulated tests, not physical multi-GPU/USB receiver certification. GPU memory, driver, frame resolution and enhancement choice affect performance; no universal real-time guarantee is made.