#!/usr/bin/env python3
"""
Prediction Script for Cotton-Weed Detection Challenge

Script to generate predictions and create Kaggle submission CSV.
Just modify the configuration section and run!

Usage:
    python predict.py

Submission Format:
    CSV with: image_id, prediction_string
    Prediction: "class conf x y w h" (space-separated)
    No detections: "no box"

Learn more: See README.md for project documentation
"""

from pathlib import Path
import pandas as pd
import shutil
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

# Model weights path (from training)
MODEL_WEIGHTS = "runs/detect/yolov8n_baseline4/weights/best.pt"

# Inference settings
CONFIDENCE_THRESHOLD = 0  # Confidence threshold for detections
IMAGE_SIZE = 640  # Input image size (FIXED by competition)
DEVICE = 0  # GPU device (0 for first GPU, 'cpu' for CPU)

# Output
OUTPUT_CSV = "submission.csv"  # Output submission file

# ============================================================================
# PREDICTION PIPELINE - No need to edit below this line
# ============================================================================


def main():
    """Generate predictions and create submission CSV."""
    print("=" * 70)
    print("COTTON WEED DETECTION - PREDICTIONS")
    print("=" * 70)

    # Verify model weights exist
    weights_path = Path(MODEL_WEIGHTS)
    if not weights_path.exists():
        print(f"\n!!! ERROR: Model weights not found: {weights_path}")
        print("\n Make sure you:")
        print("   1. Trained a model first (run train_standard.py)")
        print("   2. Set the correct path in MODEL_WEIGHTS")
        return

    # Verify test directory exists
    test_dir = Path("test/images")
    if not test_dir.exists():
        print(f"\n!!! ERROR: Test directory not found: {test_dir}")
        print("   Expected: test/images/")
        return

    # Count test images
    test_images = list(test_dir.glob("*.jpg"))
    if not test_images:
        print(f"\n!!! ERROR: No test images found in {test_dir}")
        return

    print(f"\n Test images: {len(test_images)}")
    print(f" Model: {weights_path}")
    print(f"  Confidence: {CONFIDENCE_THRESHOLD}")

    # Load model
    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)
    model = YOLO(str(weights_path))
    print("OK - Model loaded")

    # Clean up existing predictions
    pred_dir = Path("predictions")
    if pred_dir.exists():
        shutil.rmtree(pred_dir)

    # Run inference
    print("\n" + "=" * 70)
    print(" Generating Predictions")
    print("=" * 70 + "\n")

    model.predict(
        source=str(test_dir),
        save=False,  # Don't save annotated images
        save_txt=True,  # Save predictions as text
        save_conf=True,  # Include confidence scores
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        project=str(pred_dir.parent),
        name=pred_dir.name,
        exist_ok=False,
    )

    print("\nOK - Predictions generated")

    # Convert to Kaggle format
    print("\n" + "=" * 70)
    print("Creating Submission CSV")
    print("=" * 70)

    labels_dir = pred_dir / "labels"
    submission_data = []
    images_with_preds = 0
    total_boxes = 0

    for img_path in sorted(test_images, key=lambda x: x.stem):
        image_id = img_path.stem
        pred_file = labels_dir / f"{image_id}.txt"

        # Read predictions
        if pred_file.exists() and pred_file.stat().st_size > 0:
            prediction_boxes = []

            with open(pred_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        # YOLO format: class xc yc w h conf
                        # Kaggle format: class conf xc yc w h
                        class_id = parts[0]
                        conf = parts[5]
                        xc, yc, w, h = parts[1], parts[2], parts[3], parts[4]
                        prediction_boxes.append(f"{class_id} {conf} {xc} {yc} {w} {h}")
                        total_boxes += 1

            if prediction_boxes:
                prediction_string = " ".join(prediction_boxes)
                images_with_preds += 1
            else:
                prediction_string = "no box"
        else:
            prediction_string = "no box"

        submission_data.append(
            {"image_id": image_id, "prediction_string": prediction_string}
        )

    # Create DataFrame and save
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(OUTPUT_CSV, index=False)

    # Summary
    print("\n Statistics:")
    print(f"   Total images: {len(submission_df)}")
    print(f"   Images with predictions: {images_with_preds}")
    print(f"   Images without predictions: {len(submission_df) - images_with_preds}")
    print(f"   Total bounding boxes: {total_boxes}")
    print(f"   Avg boxes/image: {total_boxes / len(submission_df):.2f}")

    # Show samples
    print("\n Sample Predictions (first 5):")
    print("-" * 70)
    pd.set_option("display.max_colwidth", 80)
    print(submission_df.head(5).to_string(index=False))

    # Cleanup
    if pred_dir.exists():
        shutil.rmtree(pred_dir)

    # Done!
    print("\n" + "=" * 70)
    print("OK - SUBMISSION READY!")
    print("=" * 70)
    print(f"\n File: {OUTPUT_CSV}")
    print(f" Model: {weights_path}")
    print(f"  Confidence: {CONFIDENCE_THRESHOLD}")

    print("\n Next Steps:")
    print(f"   1. Upload '{OUTPUT_CSV}' to Kaggle")
    print("   2. Check your leaderboard score")
    print("   3. Use SafeDNN-Clean to analyze and fix data issues")
    print("   4. Retrain with cleaned data!")
    print("\n You have 3 submissions per day - use them wisely!")


if __name__ == "__main__":
    main()
