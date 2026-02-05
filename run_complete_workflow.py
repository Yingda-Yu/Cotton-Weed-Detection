#!/usr/bin/env python3
"""
Full data cleaning and training workflow.
Runs: baseline training -> data cleaning -> cleaned training -> performance comparison.

Usage:
    python run_complete_workflow.py
"""

import subprocess
import sys
import time
from pathlib import Path
import json

# Config
EPOCHS = 30
BATCH_SIZE = 16
BASELINE_NAME = "yolov8n_baseline_fast2"
CLEANED_NAME = "yolov8n_cleaned_fast"
BASELINE_MODEL = f"runs/detect/{BASELINE_NAME}/weights/best.pt"


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def wait_for_training_complete(model_path, max_wait=3600):
    """Wait until training produces the model file."""
    print(f"\nWaiting for training: {model_path}")
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if Path(model_path).exists():
            print("Training complete.")
            return True
        time.sleep(10)
        print(".", end="", flush=True)
    print("\nTimeout: training may still be running.")
    return False


def step1_train_baseline():
    """Step 1: Train baseline model."""
    print_section("Step 1: Train baseline model")

    print("Config:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print("  Data: dataset.yaml (original train)")
    print(f"  Output: {BASELINE_NAME}")

    if Path(BASELINE_MODEL).exists():
        print(f"\nBaseline already exists: {BASELINE_MODEL}")
        print("Skipping baseline training.")
        return True

    print("\nStarting baseline training...")
    cmd = [
        sys.executable,
        "train_standard.py",
        "--data", "dataset.yaml",
        "--epochs", str(EPOCHS),
        "--batch", str(BATCH_SIZE),
        "--imgsz", "640",
        "--device", "0",
        "--workers", "4",
        "--name", BASELINE_NAME
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Baseline training failed.")
        return False
    if Path(BASELINE_MODEL).exists():
        print(f"Baseline done: {BASELINE_MODEL}")
        return True
    print("Training finished but model file not found.")
    return False


def step2_analyze_quality():
    """Step 2: Analyze training set label quality."""
    print_section("Step 2: Label quality analysis")

    if not Path(BASELINE_MODEL).exists():
        print(f"Baseline model not found: {BASELINE_MODEL}")
        return False

    print(f"Model: {BASELINE_MODEL}")
    print("Split: train")

    cmd = [
        sys.executable,
        "tools/run_label_quality_analysis.py",
        "--model", BASELINE_MODEL,
        "--split", "train"
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Label quality analysis failed.")
        return False

    quality_report = "quality_report_train.json"
    if Path(quality_report).exists():
        print(f"Quality analysis done: {quality_report}")
        return True
    print("Analysis finished but report not found.")
    return False


def step3_clean_dataset():
    """Step 3: Clean training set annotations."""
    print_section("Step 3: Clean training set")

    quality_report = "quality_report_train.json"
    predictions_file = "predictions_train_coco.json"

    if not Path(quality_report).exists():
        print(f"Quality report not found: {quality_report}. Run step 2 first.")
        return False
    if not Path(predictions_file).exists():
        print(f"Predictions not found: {predictions_file}. Run step 2 first.")
        return False

    print(f"Quality report: {quality_report}")
    print(f"Predictions: {predictions_file}")

    cmd = [
        sys.executable,
        "tools/clean_dataset.py",
        "--quality-report", quality_report,
        "--predictions", predictions_file,
        "--output", "cleaned_train_annotations.json"
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Data cleaning failed.")
        return False

    cleaned_file = "cleaned_train_annotations.json"
    if Path(cleaned_file).exists():
        print(f"Cleaning done: {cleaned_file}")
        return True
    print("Cleaning finished but output not found.")
    return False


def step4_convert_and_prepare():
    """Step 4: Convert format and prepare cleaned dataset."""
    print_section("Step 4: Prepare cleaned dataset")

    print("Converting format and preparing files...")
    try:
        from tools.run_cleaning_and_comparison import (
            step2_convert_to_yolo,
            step3_copy_images,
            step4_create_dataset_yaml
        )

        print("\n[4.1] Convert to YOLO format...")
        labels_dir = step2_convert_to_yolo()
        if not labels_dir:
            return False

        print("\n[4.2] Copy images...")
        if not step3_copy_images():
            return False

        print("\n[4.3] Create dataset YAML...")
        yaml_file = step4_create_dataset_yaml()
        if not yaml_file:
            return False

        print(f"Dataset ready: {yaml_file}")
        return True
    except Exception as e:
        print(f"Prepare dataset failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step5_train_cleaned():
    """Step 5: Train on cleaned data."""
    print_section("Step 5: Train on cleaned data")

    dataset_yaml = "dataset_cleaned.yaml"
    if not Path(dataset_yaml).exists():
        print(f"Dataset config not found: {dataset_yaml}. Run step 4 first.")
        return False

    print("Config:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Data: {dataset_yaml} (cleaned train)")
    print(f"  Output: {CLEANED_NAME}")

    print("\nStarting training on cleaned data...")
    cmd = [
        sys.executable,
        "train_standard.py",
        "--data", dataset_yaml,
        "--epochs", str(EPOCHS),
        "--batch", str(BATCH_SIZE),
        "--imgsz", "640",
        "--device", "0",
        "--workers", "4",
        "--name", CLEANED_NAME
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Cleaned training failed.")
        return False

    cleaned_model = f"runs/detect/{CLEANED_NAME}/weights/best.pt"
    if Path(cleaned_model).exists():
        print(f"Cleaned training done: {cleaned_model}")
        return True
    print("Training finished but model file not found.")
    return False


def step6_compare_performance():
    """Step 6: Compare baseline vs cleaned performance."""
    print_section("Step 6: Performance comparison")

    baseline_model = BASELINE_MODEL
    cleaned_model = f"runs/detect/{CLEANED_NAME}/weights/best.pt"

    baseline_results = Path(baseline_model).parent.parent / "results.csv"
    cleaned_results = Path(cleaned_model).parent.parent / "results.csv"

    baseline_map = None
    cleaned_map = None

    if baseline_results.exists():
        try:
            import pandas as pd
            df = pd.read_csv(baseline_results)
            if len(df) > 0:
                last_row = df.iloc[-1]
                for col in df.columns:
                    if 'map50' in col.lower() and 'metrics' in col.lower():
                        baseline_map = last_row.get(col, None)
                        break
        except Exception:
            pass

    if cleaned_results.exists():
        try:
            import pandas as pd
            df = pd.read_csv(cleaned_results)
            if len(df) > 0:
                last_row = df.iloc[-1]
                for col in df.columns:
                    if 'map50' in col.lower() and 'metrics' in col.lower():
                        cleaned_map = last_row.get(col, None)
                        break
        except Exception:
            pass

    print("\nPerformance:")
    print(f"  Baseline: {baseline_model}")
    if baseline_map is not None:
        print(f"    mAP@0.5: {baseline_map:.4f} ({baseline_map*100:.2f}%)")
    else:
        print("    mAP@0.5: (not found)")

    print(f"\n  Cleaned: {cleaned_model}")
    if cleaned_map is not None:
        print(f"    mAP@0.5: {cleaned_map:.4f} ({cleaned_map*100:.2f}%)")
    else:
        print("    mAP@0.5: (not found)")

    if baseline_map is not None and cleaned_map is not None:
        improvement = cleaned_map - baseline_map
        improvement_pct = (improvement / baseline_map * 100) if baseline_map > 0 else 0
        print("\n  Improvement:")
        print(f"    Absolute: {improvement:+.4f} ({improvement*100:+.2f}%)")
        print(f"    Relative: {improvement_pct:+.2f}%")

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "baseline": {"model": str(baseline_model), "mAP50": float(baseline_map)},
            "cleaned": {"model": str(cleaned_model), "mAP50": float(cleaned_map)},
            "improvement": {"absolute": float(improvement), "percentage": float(improvement_pct)}
        }
        report_file = "complete_workflow_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved: {report_file}")

    return True


def main():
    """Run full workflow."""
    print("=" * 70)
    print("Full data cleaning and training workflow")
    print("=" * 70)
    print("\nConfig:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Baseline name: {BASELINE_NAME}")
    print(f"  Cleaned name: {CLEANED_NAME}")

    steps = [
        ("Train baseline", step1_train_baseline),
        ("Analyze label quality", step2_analyze_quality),
        ("Clean dataset", step3_clean_dataset),
        ("Prepare cleaned dataset", step4_convert_and_prepare),
        ("Train on cleaned data", step5_train_cleaned),
        ("Compare performance", step6_compare_performance),
    ]

    for i, (name, func) in enumerate(steps, 1):
        print(f"\n{'='*70}")
        print(f"Step {i}/{len(steps)}: {name}")
        print(f"{'='*70}")
        if not func():
            print(f"\nStep {i} failed. Aborting.")
            return False
        print(f"\nStep {i} done.")

    print("\n" + "=" * 70)
    print("Workflow completed successfully.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
