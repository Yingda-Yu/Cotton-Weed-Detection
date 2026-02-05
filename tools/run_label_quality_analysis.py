#!/usr/bin/env python3
"""
Full label quality analysis pipeline.
Combines YOLO prediction, COCO conversion, and SafeDNN-Clean analysis.

Usage:
    python run_label_quality_analysis.py \
        --model runs/detect/yolov8n_baseline/weights/best.pt \
        --split val
"""

import subprocess
import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_analysis_pipeline(
    model_weights,
    split="val",
    iou_threshold=0.5,
    quality_threshold=0.5,
    conf_threshold=0.25
):
    """
    Run full label quality analysis pipeline.

    Args:
        model_weights: Path to model weights
        split: Dataset split (train or val)
        iou_threshold: IoU clustering threshold
        quality_threshold: Quality score threshold
        conf_threshold: Prediction confidence threshold
    """
    print("=" * 70)
    print("Label quality analysis (SafeDNN-Clean)")
    print("=" * 70)

    safednn_script = Path("otherwork/safednn-clean/safednn-clean.py")
    if not safednn_script.exists():
        print(f"\nError: SafeDNN-Clean script not found: {safednn_script}")
        print("  Ensure otherwork/safednn-clean/safednn-clean.py exists.")
        return False

    annotations_file = f"annotations_{split}_coco.json"
    predictions_file = f"predictions_{split}_coco.json"
    quality_report_file = f"quality_report_{split}.json"

    # Step 1: Convert annotations to COCO
    print(f"\n[1/4] Convert {split} annotations to COCO...")
    try:
        from dataset.yolo_to_coco import yolo_to_coco
        yolo_to_coco(split, annotations_file)
    except Exception as e:
        print(f"Error: Convert annotations failed: {e}")
        return False

    # Step 2: Generate predictions (COCO format)
    print(f"\n[2/4] Generate model predictions...")
    try:
        from dataset.generate_predictions_coco import generate_predictions_coco
        generate_predictions_coco(
            model_weights,
            split,
            annotations_file,
            predictions_file,
            conf_threshold
        )
    except Exception as e:
        print(f"Error: Generate predictions failed: {e}")
        return False

    # Step 3: Run SafeDNN-Clean
    print(f"\n[3/4] Run SafeDNN-Clean...")
    print(f"   IoU threshold: {iou_threshold}")
    print(f"   Quality threshold: {quality_threshold}")
    try:
        subprocess.run([
            sys.executable,
            str(safednn_script),
            "--iou", str(iou_threshold),
            "--threshold", str(quality_threshold),
            "-o", quality_report_file,
            annotations_file,
            predictions_file
        ], capture_output=True, text=True, check=True)
        print("SafeDNN-Clean done.")
    except subprocess.CalledProcessError as e:
        print("Error: SafeDNN-Clean failed")
        print(f"   Return code: {e.returncode}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False

    # Step 4: Summarize results
    print(f"\n[4/4] Summarize results...")
    try:
        with open(quality_report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)

        issues = {"spurious": 0, "missing": 0, "location": 0, "label": 0}
        quality_scores = []
        for ann in report["annotations"]:
            if "issue" in ann:
                issue_type = ann["issue"]
                if issue_type in issues:
                    issues[issue_type] += 1
            if "quality" in ann:
                quality_scores.append(ann["quality"])

        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"  Total annotations: {len(report['annotations'])}")
        print(f"  With issues: {sum(issues.values())}")
        print(f"\n  Issue distribution:")
        print(f"    Spurious: {issues['spurious']} (annotated but not detected)")
        print(f"    Missing: {issues['missing']} (detected but not annotated)")
        print(f"    Location: {issues['location']} (wrong bbox)")
        print(f"    Label: {issues['label']} (wrong class)")

        if quality_scores:
            print(f"\n  Quality scores:")
            print(f"    Min: {min(quality_scores):.3f}")
            print(f"    Max: {max(quality_scores):.3f}")
            print(f"    Mean: {sum(quality_scores)/len(quality_scores):.3f}")
            print(f"    Median: {sorted(quality_scores)[len(quality_scores)//2]:.3f}")

        print(f"\n  Report: {quality_report_file}")
        print("=" * 70)

        print("\nSuggestions:")
        print("  1. Visualize: python visualize_quality_report.py ...")
        print("  2. Sort by quality and fix low-quality first")
        print("  3. By issue: spurious=remove, missing=add, location=adjust bbox, label=fix class")
        return True
    except Exception as e:
        print(f"Error: Summarize failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run full label quality analysis")
    parser.add_argument("--model", type=str, required=True, help="Model weights path")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="val",
                        help="Dataset split (train or val)")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU clustering threshold (default: 0.5)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Quality score threshold (default: 0.5)")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold (default: 0.25)")
    args = parser.parse_args()

    success = run_analysis_pipeline(
        args.model, args.split, args.iou, args.threshold, args.conf
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
