"""Create a self-contained AURA environment without modifying ML/.venv."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def run(command):
    print('> ' + ' '.join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], check=True, cwd=ROOT)


def nvidia_devices():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def choose_runtime(device, channel, names):
    use_cuda = device == 'cuda' or (device == 'auto' and bool(names))
    if not use_cuda:
        if channel != 'auto':
            raise ValueError('A CUDA channel requires --device cuda, or a detected NVIDIA GPU.')
        return 'cpu', '2.5.1', '0.20.1'
    selected = channel if channel != 'auto' else ('cu128' if re.search(r'RTX\s*50\d\d', names, re.I) else 'cu121')
    return (selected, '2.7.1', '0.22.1') if selected == 'cu128' else (selected, '2.5.1', '0.20.1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--cuda-channel', choices=['auto', 'cu121', 'cu128'], default='auto')
    parser.add_argument('--plan', action='store_true', help='Show installation commands without changing anything')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11) or sys.maxsize <= 2**32:
        raise SystemExit('Use 64-bit Python 3.11: py -3.11 setup_aura.py')
    names = nvidia_devices()
    channel, torch_version, vision_version = choose_runtime(args.device, args.cuda_channel, names)
    environment = ROOT / 'ML/.venv-portable'
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    commands = [
        [python, '-m', 'pip', 'install', '--upgrade', 'pip'],
        [python, '-m', 'pip', 'install', f'torch=={torch_version}', f'torchvision=={vision_version}',
         '--index-url', f'https://download.pytorch.org/whl/{channel}'],
        [python, '-m', 'pip', 'install', '-r', ROOT / 'ML/requirements.txt'],
        [python, '-m', 'pip', 'check'],
        [python, ROOT / 'ML/verify_setup.py', '--device', 'cpu' if channel == 'cpu' else 'cuda'],
    ]
    print('Detected NVIDIA devices: ' + (names or 'none; CPU mode'), flush=True)
    print(f'Install target: {environment}\nPyTorch runtime: {channel}. Internet is needed for package downloads.', flush=True)
    if args.plan:
        for command in commands:
            print(subprocess.list2cmdline([str(part) for part in command]))
        return
    # Fail early on an incomplete copy, before downloading large Python packages.
    run([sys.executable, ROOT / 'ML/verify_setup.py', '--assets-only'])
    if environment.exists() and not (environment / 'pyvenv.cfg').is_file():
        raise SystemExit('ML/.venv-portable exists but is not a virtual environment. Choose a clean project folder.')
    venv.EnvBuilder(with_pip=True).create(environment)
    for command in commands:
        run(command)
    (environment / 'aura-setup.json').write_text(json.dumps(dict(channel=channel, torch=torch_version,
        torchvision=vision_version, python=sys.version), indent=2), encoding='utf-8')
    print('\nAURA setup and model checks passed. Start AURA.cmd opens the dashboard.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'\nSetup failed: {error}\nSee README.md. Existing ML/.venv was not modified.', file=sys.stderr)
        sys.exit(1)
