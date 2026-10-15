"""Repackage an extracted PyTorch archive without unpickling its contents."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def recover(source: Path, destination: Path):
    required = ['data.pkl', 'version', 'byteorder']
    if any(not (source / name).is_file() for name in required):
        raise ValueError('Source is not an extracted PyTorch checkpoint')
    tensors = list((source / 'data').iterdir())
    if not tensors:
        raise ValueError('No tensor storage files found')
    if destination.exists():
        raise FileExistsError(f'Will not overwrite {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_STORED) as archive:
        for file in sorted(source.rglob('*')):
            if file.is_file():
                archive.write(file, 'best/' + file.relative_to(source).as_posix())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('Recovered archive failed CRC validation')
        for file in source.rglob('*'):
            if file.is_file():
                assert archive.read('best/' + file.relative_to(source).as_posix()) == file.read_bytes()
    return {'path': str(destination.resolve()), 'bytes': destination.stat().st_size,
            'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'tensor_files': len(tensors), 'archive_verified': True}


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root / 'models/cbam_best/best')
    parser.add_argument('--output', type=Path, default=root / 'models/yolov8m_visdrone_recovered.pt')
    args = parser.parse_args()
    print(json.dumps(recover(args.source, args.output), indent=2))
