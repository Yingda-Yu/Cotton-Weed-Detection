#!/usr/bin/env python3
"""
Full data cleaning and performance comparison pipeline.

Runs:
1. Clean dataset annotations
2. Convert back to YOLO format
3. Copy image files
4. Train model on cleaned data
5. Compare baseline vs cleaned performance

Note: Original dataset is never modified.
"""

import json
import yaml
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse

# ============================================================================
# Config
# ============================================================================

QUALITY_REPORT = "quality_report_train.json"
PREDICTIONS_FILE = "predictions_train_coco.json"
CLEANED_ANNOTATIONS = "cleaned_train_annotations.json"

LOCATION_THRESHOLD = 0.7
LABEL_THRESHOLD = 0.8
MISSING_THRESHOLD = 0.5

ORIGINAL_TRAIN_DIR = "train"
CLEANED_TRAIN_DIR = "cleaned_train"

BASELINE_MODEL = "runs/detect/yolov8n_baseline_new/weights/best.pt"
EPOCHS = 10
BATCH_SIZE = 8
RUN_NAME_CLEANED = "yolov8n_cleaned_new"

COMPARISON_REPORT = "cleaning_comparison_report.json"


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def step1_clean_dataset():
    """Step 1: Clean dataset annotations."""
    print_section("Step 1: Auto-clean dataset annotations")

    if not Path(QUALITY_REPORT).exists():
        print(f"Error: Quality report not found: {QUALITY_REPORT}")
        print("  Run first: python run_label_quality_analysis.py --model <model_path> --split train")
        return False

    if not Path(PREDICTIONS_FILE).exists():
        print(f"Error: Predictions not found: {PREDICTIONS_FILE}")
        print("  Run first: python run_label_quality_analysis.py --model <model_path> --split train")
        return False

    print("Running cleaning script...")
    print(f"  Quality report: {QUALITY_REPORT}")
    print(f"  Predictions: {PREDICTIONS_FILE}")
    print(f"  Output: {CLEANED_ANNOTATIONS}")
    print(f"\n  Thresholds: Location={LOCATION_THRESHOLD}, Label={LABEL_THRESHOLD}, Missing={MISSING_THRESHOLD}")
    
    try:
        from tools.clean_dataset import clean_dataset
        
        cleaned_data = clean_dataset(
            quality_report_file=QUALITY_REPORT,
            predictions_file=PREDICTIONS_FILE,
            output_file=CLEANED_ANNOTATIONS,
            location_score_threshold=LOCATION_THRESHOLD,
            label_score_threshold=LABEL_THRESHOLD,
            missing_score_threshold=MISSING_THRESHOLD
        )
        
        original_count = len([ann for ann in json.load(open(QUALITY_REPORT))["annotations"]
                             if ann.get("id", 0) >= 0])
        cleaned_count = len(cleaned_data["annotations"])
        print(f"\nCleaning done.")
        print(f"  Original annotations: {original_count}")
        print(f"  Cleaned annotations: {cleaned_count}")
        print(f"  Net change: {cleaned_count - original_count}")
        
        return {
            "original_annotations": original_count,
            "cleaned_annotations": cleaned_count,
            "net_change": cleaned_count - original_count
        }
        
    except Exception as e:
        print(f"Cleaning failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def step2_convert_to_yolo():
    """Step 2: Convert cleaned annotations to YOLO format."""
    print_section("Step 2: Convert cleaned annotations to YOLO")

    if not Path(CLEANED_ANNOTATIONS).exists():
        print(f"Error: Cleaned annotations not found: {CLEANED_ANNOTATIONS}")
        return False

    try:
        from dataset.coco_to_yolo import coco_to_yolo
        output_labels_dir = coco_to_yolo(
            coco_file=CLEANED_ANNOTATIONS,
            split_dir=ORIGINAL_TRAIN_DIR,
            output_dir=CLEANED_TRAIN_DIR
        )
        label_files = list(output_labels_dir.glob("*.txt"))
        print(f"\nConversion done. Label files: {len(label_files)}, output: {output_labels_dir.absolute()}")
        return True
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step3_copy_images():
    """Step 3: Copy image files to cleaned dataset dir."""
    print_section("Step 3: Copy images to cleaned dataset")

    original_images_dir = Path(ORIGINAL_TRAIN_DIR) / "images"
    cleaned_images_dir = Path(CLEANED_TRAIN_DIR) / "images"

    if not original_images_dir.exists():
        print(f"Error: Original images dir not found: {original_images_dir}")
        return False

    cleaned_images_dir.mkdir(parents=True, exist_ok=True)
    image_files = list(original_images_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} images.")

    copied = 0
    for img_file in image_files:
        dest = cleaned_images_dir / img_file.name
        if not dest.exists():
            shutil.copy2(img_file, dest)
            copied += 1

    print(f"\nCopy done. Copied: {copied}, output: {cleaned_images_dir.absolute()}")
    return True


def step4_create_dataset_yaml():
    """Step 4: Create dataset YAML for cleaned train (val unchanged)."""
    print_section("Step 4: Create cleaned dataset config")

    with open("dataset.yaml", 'r', encoding='utf-8') as f:
        original_config = yaml.safe_load(f)

    cleaned_config = original_config.copy()
    cleaned_config["train"] = f"{CLEANED_TRAIN_DIR}/images"
    cleaned_config["val"] = "cotton weed dataset/val/images"

    cleaned_yaml = "dataset_cleaned.yaml"
    with open(cleaned_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(cleaned_config, f, default_flow_style=False, allow_unicode=True)

    print(f"Config created: {cleaned_yaml} (train=cleaned, val=original)")
    return cleaned_yaml


def step5_train_with_cleaned_data(dataset_yaml):
    """Step 5: Train model on cleaned data."""
    print_section("Step 5: Train on cleaned data")

    print(f"Config: data={dataset_yaml}, epochs={EPOCHS}, batch={BATCH_SIZE}, run={RUN_NAME_CLEANED}")

    weights_path = Path(f"runs/detect/{RUN_NAME_CLEANED}/weights/best.pt")
    if weights_path.exists():
        print(f"\nExisting run found: {weights_path}")
        response = input("Skip training and use existing model? (y/n): ").strip().lower()
        if response == 'y':
            print("Using existing model.")
            return str(weights_path)

    try:
        from ultralytics import YOLO
        print("\nStarting training...")
        model = YOLO("yolov8n.pt")
        
        results = model.train(
            data=dataset_yaml,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=640,
            name=RUN_NAME_CLEANED,
            project="runs/detect",
            device=0,
            workers=4
        )
        
        if weights_path.exists():
            print(f"\nTraining done. Weights: {weights_path.absolute()}")
            best_map = None
            try:
                if hasattr(results, 'results_dict'):
                    best_map = results.results_dict.get('metrics/mAP50(B)', None)
                if best_map is None:
                    results_csv = weights_path.parent.parent / "results.csv"
                    if results_csv.exists():
                        try:
                            import pandas as pd
                            df = pd.read_csv(results_csv)
                            if len(df) > 0:
                                last_row = df.iloc[-1]
                                for col in df.columns:
                                    if 'map50' in col.lower() or 'mAP50' in col:
                                        best_map = last_row.get(col, None)
                                        break
                        except Exception:
                            with open(results_csv, 'r') as f:
                                lines = f.readlines()
                                if len(lines) > 1:
                                    headers = lines[0].strip().split(',')
                                    last_line = lines[-1].strip().split(',')
                                    for i, h in enumerate(headers):
                                        if 'map50' in h.lower() or 'mAP50' in h:
                                            try:
                                                best_map = float(last_line[i])
                                            except:
                                                pass
                                            break
            except Exception as e:
                print(f"   Warning: Could not auto-extract mAP: {e}")

            if best_map is not None:
                print(f"   Best mAP@0.5: {best_map:.4f}")
            else:
                print("   Could not auto-extract mAP. Check training logs.")
                user_input = input("   Enter best mAP@0.5 (or Enter to skip): ").strip()
                if user_input:
                    try:
                        best_map = float(user_input)
                    except ValueError:
                        best_map = None
            
            return {
                "weights_path": str(weights_path),
                "best_map": best_map,
                "epochs": EPOCHS
            }
        else:
            print(f"Error: Training finished but weights not found: {weights_path}")
            return None

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def step6_get_baseline_performance():
    """Step 6: Get baseline model performance."""
    print_section("Step 6: Get baseline performance")

    baseline_weights = Path(BASELINE_MODEL)
    if not baseline_weights.exists():
        print(f"Warning: Baseline model not found: {BASELINE_MODEL}")
        return None
    baseline_run_dir = baseline_weights.parent.parent
    results_file = baseline_run_dir / "results.csv"
    
    best_map = None
    
    if results_file.exists():
        try:
            try:
                import pandas as pd
                df = pd.read_csv(results_file)
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    best_map = last_row.get('metrics/mAP50(B)', None)
                    if best_map is None or pd.isna(best_map):
                        for col in df.columns:
                            if 'map50' in col.lower() or 'mAP50' in col:
                                best_map = last_row.get(col, None)
                                break
            except ImportError:
                with open(results_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        headers = lines[0].strip().split(',')
                        last_line = lines[-1].strip().split(',')
                        map_col_idx = None
                        for i, h in enumerate(headers):
                            if 'map50' in h.lower() or 'mAP50' in h:
                                map_col_idx = i
                                break
                        
                        if map_col_idx is not None and map_col_idx < len(last_line):
                            try:
                                best_map = float(last_line[map_col_idx])
                            except ValueError:
                                pass
        except Exception as e:
            print(f"Could not read results file: {e}")

    if best_map is not None:
        print(f"Baseline: {BASELINE_MODEL}, mAP@0.5: {best_map:.4f}")
        return {"weights_path": str(baseline_weights), "best_map": float(best_map)}
    else:
        print("Could not get baseline performance automatically.")
        user_input = input("Baseline mAP@0.5 (Enter to skip): ").strip()
        if user_input:
            try:
                best_map = float(user_input)
                return {"weights_path": str(baseline_weights), "best_map": best_map, "source": "manual_input"}
            except ValueError:
                print("Invalid input.")
        return None


def step7_compare_performance(baseline_perf, cleaned_perf, cleaning_stats):
    """Step 7: Compare baseline vs cleaned performance."""
    print_section("Step 7: Performance comparison")
    
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "cleaning_stats": cleaning_stats,
        "baseline": baseline_perf,
        "cleaned": cleaned_perf
    }
    
    if baseline_perf and cleaned_perf:
        baseline_map = baseline_perf.get("best_map", 0)
        cleaned_map = cleaned_perf.get("best_map", 0)
        
        improvement = cleaned_map - baseline_map
        improvement_pct = (improvement / baseline_map * 100) if baseline_map > 0 else 0
        
        comparison["improvement"] = {
            "absolute": improvement,
            "percentage": improvement_pct
        }
        
        print(f"\nPerformance: Baseline mAP@0.5 {baseline_map:.4f}, Cleaned {cleaned_map:.4f}")
        print(f"  Absolute: {improvement:+.4f}, Relative: {improvement_pct:+.2f}%")
        if improvement > 0:
            print("  Cleaned model improved.")
        elif improvement < 0:
            print("  Cleaned model decreased; consider adjusting thresholds.")
        else:
            print("  No significant change.")

    with open(COMPARISON_REPORT, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {COMPARISON_REPORT}")
    
    return comparison


def main():
    """Main pipeline."""
    global BASELINE_MODEL, EPOCHS, LOCATION_THRESHOLD, LABEL_THRESHOLD, MISSING_THRESHOLD

    parser = argparse.ArgumentParser(description="Full data cleaning and performance comparison")
    parser.add_argument("--skip-cleaning", action="store_true", help="Skip cleaning if already done")
    parser.add_argument("--skip-training", action="store_true", help="Skip training if already done")
    parser.add_argument("--baseline-model", type=str, default=BASELINE_MODEL, help="Baseline model path")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs")
    parser.add_argument("--location-threshold", type=float, default=LOCATION_THRESHOLD, help="Location fix threshold")
    parser.add_argument("--label-threshold", type=float, default=LABEL_THRESHOLD, help="Label fix threshold")
    parser.add_argument("--missing-threshold", type=float, default=MISSING_THRESHOLD, help="Missing add threshold")
    args = parser.parse_args()

    BASELINE_MODEL = args.baseline_model
    EPOCHS = args.epochs
    LOCATION_THRESHOLD = args.location_threshold
    LABEL_THRESHOLD = args.label_threshold
    MISSING_THRESHOLD = args.missing_threshold

    print("=" * 70)
    print("  Full data cleaning and performance comparison")
    print("=" * 70)
    print(f"\nConfig: baseline={BASELINE_MODEL}, epochs={EPOCHS}, thresholds L/L/M={LOCATION_THRESHOLD}/{LABEL_THRESHOLD}/{MISSING_THRESHOLD}, output={CLEANED_TRAIN_DIR}")
    
    results = {}
    
    if not args.skip_cleaning:
        cleaning_stats = step1_clean_dataset()
        if cleaning_stats is None:
            print("\nCleaning failed. Aborting.")
            return
        results["cleaning_stats"] = cleaning_stats
    else:
        print("\nSkipping cleaning step.")
        if Path(CLEANED_ANNOTATIONS).exists():
            with open(CLEANED_ANNOTATIONS, 'r', encoding='utf-8') as f:
                cleaned_data = json.load(f)
            results["cleaning_stats"] = {
                "cleaned_annotations": len(cleaned_data["annotations"])
            }
    
    if not Path(CLEANED_TRAIN_DIR).exists() or not list(Path(CLEANED_TRAIN_DIR).glob("labels/*.txt")):
        if not step2_convert_to_yolo():
            print("\nConversion failed. Aborting.")
            return
    else:
        print("\nSkipping conversion (cleaned labels exist).")

    cleaned_images_dir = Path(CLEANED_TRAIN_DIR) / "images"
    if not cleaned_images_dir.exists() or not list(cleaned_images_dir.glob("*.jpg")):
        if not step3_copy_images():
            print("\nCopy images failed. Aborting.")
            return
    else:
        print("\nSkipping copy images (already exist).")

    dataset_yaml = step4_create_dataset_yaml()

    if not args.skip_training:
        cleaned_perf = step5_train_with_cleaned_data(dataset_yaml)
        if cleaned_perf is None:
            print("\nTraining failed. Aborting.")
            return
        results["cleaned_performance"] = cleaned_perf
    else:
        print("\nSkipping training.")
        weights_path = Path(f"runs/detect/{RUN_NAME_CLEANED}/weights/best.pt")
        if weights_path.exists():
            results["cleaned_performance"] = {"weights_path": str(weights_path), "note": "existing model"}

    baseline_perf = step6_get_baseline_performance()
    if baseline_perf:
        results["baseline_performance"] = baseline_perf

    comparison = step7_compare_performance(
        baseline_perf,
        results.get("cleaned_performance"),
        results.get("cleaning_stats")
    )
    
    print("\n" + "=" * 70)
    print("  Pipeline complete.")
    print("=" * 70)
    print(f"\nOutputs: {CLEANED_ANNOTATIONS}, {CLEANED_TRAIN_DIR}/, dataset_cleaned.yaml, {COMPARISON_REPORT}")
    print(f"Train set cleaned: {CLEANED_TRAIN_DIR}/. Val set unchanged for evaluation.")
    print(f"Next: Check {COMPARISON_REPORT}; tune thresholds or inspect cleaning if needed.")
    print("=" * 70)


if __name__ == "__main__":
    main()

