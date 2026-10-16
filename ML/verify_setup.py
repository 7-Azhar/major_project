"""Check the handoff assets, device execution, and real enhancement/detection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def verify_assets():
    manifest = json.loads((PROJECT / 'assets-manifest.json').read_text(encoding='utf-8'))
    problems = []
    for item in manifest['assets']:
        path = PROJECT / item['path']
        if not path.is_file():
            problems.append('Missing: ' + item['path'])
            continue
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != item['bytes'] or digest != item['sha256']:
            problems.append('Incomplete or changed: ' + item['path'])
    if problems:
        raise RuntimeError('\n'.join(problems) + '\nDownload/clone the complete repository again, or restore the listed files.')
    print(f"Verified {len(manifest['assets'])} bundled assets (SHA-256).", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-only', action='store_true')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    verify_assets()
    if args.assets_only:
        return
    import cv2
    import torch
    from ultralytics import YOLO
    from hardware import Hardware
    from frame_processor import FrameProcessor
    from detect_video import VISDRONE_CLASSES
    sys.path.insert(0, str(PROJECT / 'feeed enhancement'))
    from app.enhancers import get_enhancer_by_name
    hardware = Hardware(torch, args.device)
    print(json.dumps(hardware.report(), indent=2), flush=True)
    model = YOLO(str(ROOT / 'models/yolov8m_visdrone_recovered.pt')).to(hardware.device)
    if list(model.names.values()) != VISDRONE_CLASSES:
        raise RuntimeError('Unexpected detector classes')
    capture = cv2.VideoCapture(str(PROJECT / 'feeed enhancement/samples/fpv/real_fpv_user_flight.mp4'))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError('The bundled demo video could not be decoded')
    processor = FrameProcessor(cv2, model, hardware, get_enhancer_by_name, ROOT / 'profiles/user_flight_osd.json')
    for method in ('identity', 'zerodce'):
        packet = processor.process(frame, dict(enhancement=method, detection='exclude25', outlines=True), new_frame=True)
        assert packet['output'] and packet['original']
        print(f'{method}: enhancement + VisDrone detection passed.', flush=True)
    print('Ready. Open Start AURA.cmd, then choose your video or camera.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'AURA verification failed: {error}', file=sys.stderr)
        sys.exit(1)
