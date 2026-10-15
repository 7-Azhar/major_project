# Running AURA on another computer

AURA selects hardware through PyTorch, not an RTX model-name check. The main dashboard now supports `auto`, `cpu`, `cuda`, and `cuda:N`. Auto selects GPU 0 when CUDA is available, otherwise a visibly identified CPU path. CPU processing is slower. Explicit CUDA requests fail clearly if unavailable. A small startup execution check catches some driver/runtime incompatibilities before loading the detector. It is not a guarantee that every model operation or video size fits memory.

RTX 2050 defaults are unchanged: GPU 0, FP16, input size 640, the same pretrained detector, enhancement parameters and exclusion modes. The existing environment and active dashboard are not modified by these instructions.

## Copy project assets, not the virtual environment

Copy the `ML` code and `feeed enhancement/app` folder with their structure intact. Include:

- `ML/models/yolov8m_visdrone_recovered.pt`
- `ML/profiles/user_flight_osd.json`
- `feeed enhancement/outputs/models/Epoch99.pth`
- `feeed enhancement/samples/fpv/real_fpv_user_flight.mp4` for the default recording
- `ML/dashboard` and the launch scripts

Model files are ignored by Git, so cloning code alone does not include them. Do not copy `.venv`: the existing laptop environment has absolute links to another local installation. Create a fresh environment on the destination machine. These steps are for Windows with Python 3.11 installed.

## Fresh setup on the destination PC

From the copied project root:

```powershell
py -3.11 -m venv ML\.venv-portable
& .\ML\.venv-portable\Scripts\python.exe -m pip install --upgrade pip
```

Use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) for a CUDA-enabled torch/torchvision pair compatible with the destination NVIDIA driver and GPU. Run its pip command through `.venv-portable\Scripts\python.exe -m pip`, not global pip. For example, if the selector recommends CUDA 12.8:

```powershell
& .\ML\.venv-portable\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& .\ML\.venv-portable\Scripts\python.exe -m pip install -r .\ML\requirements.txt
& .\ML\.venv-portable\Scripts\python.exe .\ML\hardware.py --device cuda
```

The last command must identify the GPU and pass actual tensor execution. If it fails, resolve the driver/PyTorch mismatch; copying another machine's environment will not resolve it. New GPU generations may need newer PyTorch builds. No driver installation or system changes are performed by AURA.

Launch:

```powershell
.\ML\run_dashboard.ps1 -Port 8766 -PythonPath .\ML\.venv-portable\Scripts\python.exe
# Optional: choose GPU 1, or full precision
.\ML\run_dashboard.ps1 -Port 8766 -PythonPath .\ML\.venv-portable\Scripts\python.exe -Device cuda:1 -Precision fp32
# Optional CPU mode
.\ML\run_dashboard.ps1 -Port 8766 -PythonPath .\ML\.venv-portable\Scripts\python.exe -Device cpu
```

Open http://127.0.0.1:8766/. Test raw detection and Zero-DCE on a short recording before using a camera. The startup log and dashboard report the detected device. GPU memory, driver, input resolution and enhancement choice affect speed; there is no universal real-time guarantee. The larger RTX 4050 has not been physically tested in this workspace. Multiple-GPU routing is unit-tested, not hardware-tested here. Legacy batch/experiment scripts retain their previous explicit CUDA/CPU behavior; this update targets the main dashboard.
