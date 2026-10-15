"""Explicit opt-in Kaggle training entry point. Never invoked by the video launcher."""
import argparse
from pathlib import Path
import shutil
from ultralytics import YOLO
from aura_cbam import CBAMTrainer, cbam_count, verify_training_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path, help='Verified VisDrone YAML path')
    parser.add_argument('--weights', required=True, type=Path, help='Existing trusted YOLOv8m checkpoint')
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/aura_cbam'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--batch', type=int, default=6)
    args = parser.parse_args()
    if not args.data.is_file() or not args.weights.is_file():
        parser.error('Dataset YAML and initial weights must already exist')
    if args.output.exists():
        parser.error('Choose a new output directory to preserve previous experiments')
    model = YOLO(str(args.weights))
    model.add_callback('on_train_start', verify_training_model)
    model.train(trainer=CBAMTrainer, data=str(args.data), epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch, device=0, optimizer='SGD',
                lr0=0.005, momentum=0.937, weight_decay=0.0005,
                warmup_epochs=5, warmup_momentum=0.8, warmup_bias_lr=0.1,
                hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1,
                scale=0.5, shear=0.0, perspective=0.0, flipud=0.5, fliplr=0.5,
                mosaic=1.0, mixup=0.4, close_mosaic=15, patience=50, cos_lr=True,
                save=True, save_period=1, val=True, project=str(args.output), name='train')
    best = Path(model.trainer.best)
    reloaded = YOLO(str(best))
    if cbam_count(reloaded.model) != 3:
        raise RuntimeError('Saved best checkpoint lost CBAM; refusing verified export')
    destination = args.output / 'aura_cbam_verified_best.pt'
    shutil.copy2(best, destination)
    print(f'Verified three CBAM blocks in {destination}')


if __name__ == '__main__':
    main()
