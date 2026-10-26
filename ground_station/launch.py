"""Launch the shared vision service and ground station, preserving existing servers."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from urllib.request import build_opener, ProxyHandler
import webbrowser

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def health(port):
    try:
        with build_opener(ProxyHandler({})).open(f'http://127.0.0.1:{port}/api/health', timeout=1) as response:
            return json.load(response)
    except Exception:
        return {}


def ensure_free(port):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))


def open_workspaces(args, config):
    if args.no_browser:
        return
    webbrowser.open(f'http://127.0.0.1:{args.port}')
    if args.open_both and config.video_backend == 'aura':
        webbrowser.open(f'http://127.0.0.1:{config.aura_port}/', new=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.integrated.json')
    parser.add_argument('--aura-python', type=Path)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--open-both', action='store_true', help='Open detection and standalone vision pages')
    args = parser.parse_args()
    if not args.config.is_file():
        raise ValueError(f'Configuration file not found: {args.config}')
    from groundstation.config import Config, config_fingerprint
    os.environ['GS_CONFIG'] = str(args.config.resolve())
    config = Config.load()
    if args.port == config.aura_port:
        raise ValueError('Ground station and AURA need different ports')
    existing = health(args.port)
    if existing.get('application') == 'AURA Ground Station':
        if existing.get('config_fingerprint') != config_fingerprint(config):
            raise ValueError('A ground station with different settings or an older version is already running on this port. Stop it first or select --port with another number.')
        print(f'Ground station already running: http://127.0.0.1:{args.port}', flush=True)
        open_workspaces(args, config)
        return
    ensure_free(args.port)
    child = None
    try:
        if config.video_backend == 'aura':
            vision = health(config.aura_port)
            if vision.get('application') != 'AURA':
                ensure_free(config.aura_port)
                candidates = [PROJECT / 'ML' / name / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
                              for name in ('.venv', '.venv-portable')]
                python = args.aura_python or next((p for p in candidates if p.is_file()), None)
                if python is None:
                    raise RuntimeError('Run Setup AURA.cmd first')
                child = subprocess.Popen([str(python), str(PROJECT / 'ML/dashboard.py'), '--port', str(config.aura_port)], cwd=PROJECT)
                deadline = time.monotonic()+120
                while health(config.aura_port).get('application') != 'AURA':
                    if child.poll() is not None:
                        raise RuntimeError('AURA did not start; see its error above')
                    if time.monotonic() > deadline:
                        raise RuntimeError('AURA startup timed out')
                    time.sleep(.5)
            else:
                print('Reusing the existing AURA service and its current input/settings.', flush=True)
        if not args.no_browser:
            def open_when_ready():
                for _ in range(100):
                    if health(args.port).get('application') == 'AURA Ground Station':
                        open_workspaces(args, config)
                        return
                    time.sleep(.2)
            threading.Thread(target=open_when_ready, daemon=True).start()
        import uvicorn
        uvicorn.run('groundstation.main:app', host='127.0.0.1', port=args.port)
    finally:
        # Only stop the vision process created by this launcher.
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f'Startup failed: {exc}', file=sys.stderr)
        sys.exit(1)
