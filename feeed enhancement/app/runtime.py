"""Optional Windows DLL discovery for the legacy ONNX viewer."""
import importlib.util
import os
from pathlib import Path
import sys

_dll_handles = []
_configured = False


def configure_runtime():
    global _configured
    if _configured:
        return
    _configured = True
    # Preserve the original laptop's optional overlay, without requiring it elsewhere.
    local_libs = Path(__file__).resolve().parents[1] / '.venv_libs'
    if local_libs.is_dir() and str(local_libs) not in sys.path:
        sys.path.insert(0, str(local_libs))
    if os.name != 'nt' or not hasattr(os, 'add_dll_directory'):
        return
    candidates = []
    spec = importlib.util.find_spec('torch')
    if spec and spec.origin:
        candidates.append(Path(spec.origin).parent / 'lib')
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        candidates.extend([Path(cuda_path) / 'bin', Path(cuda_path) / 'bin/x64'])
    for candidate in candidates:
        if candidate.is_dir():
            _dll_handles.append(os.add_dll_directory(str(candidate)))
