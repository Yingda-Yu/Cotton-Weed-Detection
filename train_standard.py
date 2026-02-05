#!/usr/bin/env python3
"""
Standard training script (no 3LC dependency).
Uses standard Ultralytics YOLO for training.

Usage:
    python train_standard.py
    python train_standard.py --data dataset.yaml --epochs 30 --batch 16
"""

import torch
from ultralytics import YOLO
import argparse
from pathlib import Path
import gc

# ============================================================================
# CONFIGURATION - Defaults (override via CLI)
# ============================================================================

# Dataset
DATASET_YAML = "dataset.yaml"  # Default: original dataset
# DATASET_YAML = "dataset_cleaned.yaml"  # Cleaned dataset

# Training hyperparameters
EPOCHS = 30
BATCH_SIZE = 16
IMAGE_SIZE = 640  # Fixed by competition
DEVICE = 0  # GPU index or 'cpu'
WORKERS = 4  # DataLoader workers

# Advanced
LR0 = 0.01  # Initial learning rate
PATIENCE = 20  # Early stopping patience

# Augmentation
USE_AUGMENTATION = False  # Mosaic, mixup, etc.

# Model
MODEL_WEIGHTS = "yolov8n.pt"
PROJECT_NAME = "runs/detect"
RUN_NAME = "yolov8n_standard"

# ============================================================================
# Training
# ============================================================================

def main():
    """Main training entry."""
    parser = argparse.ArgumentParser(description="Standard YOLOv8 training")
    parser.add_argument("--data", type=str, default=DATASET_YAML, help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE, help="Image size")
    parser.add_argument("--device", default=DEVICE, help="Device (GPU index or 'cpu')")
    parser.add_argument("--workers", type=int, default=WORKERS, help="DataLoader workers")
    parser.add_argument("--lr0", type=float, default=LR0, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience")
    parser.add_argument("--augment", action="store_true", help="Enable augmentation")
    parser.add_argument("--model", type=str, default=MODEL_WEIGHTS, help="Pretrained weights")
    parser.add_argument("--name", type=str, default=RUN_NAME, help="Run name")
    parser.add_argument("--project", type=str, default=PROJECT_NAME, help="Project directory")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")

    args = parser.parse_args()

    print("=" * 70)
    print("COTTON WEED DETECTION - Standard training (no 3LC)")
    print("=" * 70)

    # Environment
    print("\nEnvironment:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Dataset config
    dataset_path = Path(args.data)
    if not dataset_path.exists():
        print(f"\nError: Dataset config not found: {args.data}")
        print(f"  CWD: {Path.cwd()}")
        print("  Ensure the dataset YAML exists.")
        return

    print(f"\nDataset config: {args.data}")

    # Training config
    print("\n" + "=" * 70)
    print("Training config")
    print("=" * 70)
    print(f"  Run: {args.name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device: {'GPU ' + str(args.device) if args.device != 'cpu' else 'CPU'}")
    print(f"  LR0: {args.lr0}")
    print(f"  Patience: {args.patience}")
    print(f"  Augment: {'on' if args.augment else 'off'}")

    # Load model
    print("\n" + "=" * 70)
    print("Loading model")
    print("=" * 70)
    if args.resume:
        print(f"\nResuming: {args.resume}")
        model = YOLO(args.resume)
        print("Resumed from checkpoint.")
    else:
        print(f"\nLoading: {args.model}")
        model = YOLO(args.model)
        print("Loaded (YOLOv8n, ~3M params).")

    workers = 0 if args.workers == 0 else args.workers

    train_args = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": workers,
        "lr0": args.lr0,
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
        "val": True,
        "save": True,
        "plots": True,
        "verbose": True,
    }

    if workers == 0:
        print("\nSingle-process mode (workers=0) to reduce memory usage.")
        train_args["mosaic"] = 0.0

    if args.resume:
        train_args["resume"] = True

    if args.augment:
        train_args.update({
            "mosaic": 1.0,
            "mixup": 0.05,
            "copy_paste": 0.1,
        })
        print("\nAugmentation enabled.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "=" * 70)
    print("Starting training")
    print("=" * 70 + "\n")

    try:
        results = model.train(**train_args)

        print("\n" + "=" * 70)
        print("Training finished.")
        print("=" * 70)
        print(f"\nWeights:")
        print(f"  Best: {args.project}/{args.name}/weights/best.pt")
        print(f"  Last: {args.project}/{args.name}/weights/last.pt")

        if hasattr(results, 'results_dict'):
            print("\nMetrics:")
            if 'metrics/mAP50' in results.results_dict:
                print(f"  mAP@0.5: {results.results_dict['metrics/mAP50']:.4f}")
            if 'metrics/mAP50-95' in results.results_dict:
                print(f"  mAP@0.5:0.95: {results.results_dict['metrics/mAP50-95']:.4f}")

        print("\nNext:")
        print(f"  1. Results: {args.project}/{args.name}/")
        print(f"  2. Predict: python predict.py --model {args.project}/{args.name}/weights/best.pt")
        print("  3. Submit submission.csv to Kaggle.")

    except Exception as e:
        print(f"\nError during training: {e}")
        error_str = str(e).lower()
        if "memory" in error_str or "insufficient" in error_str or "allocate" in error_str:
            print("\n" + "=" * 70)
            print("Out of memory - suggestions")
            print("=" * 70)
            print(f"1. Reduce batch (current: {args.batch}): --batch 4 or --batch 2")
            print("2. Use single process: --workers 0")
            print("3. Close other memory-heavy apps")
            print("4. Increase Windows page file: Control Panel > System > Advanced > Performance > Advanced > Virtual memory")
            print("   e.g. Initial 8192MB, Max 16384MB")
            print("5. Try CPU: --device cpu")
            print("\nExample retry:")
            print(f"python train_standard.py --data {args.data} --epochs {args.epochs} --batch 4 --imgsz {args.imgsz} --device {args.device} --workers 0 --name {args.name} --resume {args.resume if args.resume else ''}")
        raise


if __name__ == "__main__":
    main()
