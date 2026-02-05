#!/usr/bin/env python3
"""
Auto-clean dataset annotations from SafeDNN-Clean quality report.
Fixes four issue types: spurious, location, label, missing.

Note: Original dataset is not modified; output is written to a new file.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def bbox_iou(bbox1, bbox2):
    """Compute IoU of two bboxes [x, y, w, h]."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0.0


def find_matching_prediction(gt_ann, predictions_by_image, iou_threshold=0.3):
    """Find best matching prediction for a GT annotation."""
    image_id = gt_ann["image_id"]
    if image_id not in predictions_by_image:
        return None
    gt_bbox = gt_ann["bbox"]
    best_match = None
    best_iou = 0.0
    for pred in predictions_by_image[image_id]:
        iou = bbox_iou(gt_bbox, pred["bbox"])
        if iou > best_iou and iou >= iou_threshold:
            best_iou = iou
            best_match = pred
    return best_match


def clean_dataset(
    quality_report_file="quality_report.json",
    predictions_file="predictions_coco.json",
    output_file="cleaned_annotations.json",
    spurious_threshold=0.3,
    location_score_threshold=0.7,
    label_score_threshold=0.8,
    missing_score_threshold=0.5
):
    """
    Clean dataset annotations from quality report and predictions.

    Args:
        quality_report_file: SafeDNN-Clean quality report path
        predictions_file: Model predictions (for location/label matching)
        output_file: Output path (original data not overwritten)
        spurious_threshold: unused; all spurious are removed
        location_score_threshold: Min score to fix location
        label_score_threshold: Min score to fix label
        missing_score_threshold: Min score to add missing
    """
    print("=" * 70)
    print("Dataset auto-cleaning")
    print("=" * 70)
    print(f"Original data is not modified. Output: {output_file}")
    print("=" * 70)

    print(f"\n[1/5] Load quality report: {quality_report_file}")
    with open(quality_report_file, 'r', encoding='utf-8') as f:
        quality_report = json.load(f)

    print(f"[2/5] Load predictions: {predictions_file}")
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)

    predictions_by_image = defaultdict(list)
    for pred in predictions_data["annotations"]:
        predictions_by_image[pred["image_id"]].append(pred)
    print(f"   Total prediction boxes: {len(predictions_data['annotations'])}")

    print(f"\n[3/5] Analyze annotations...")
    all_annotations = quality_report["annotations"]
    gt_annotations = [ann for ann in all_annotations if ann.get("id", 0) >= 0]
    missing_predictions = [ann for ann in all_annotations if ann.get("id", 0) < 0]
    print(f"   GT annotations: {len(gt_annotations)}")
    print(f"   Missing predictions: {len(missing_predictions)}")

    issue_stats = defaultdict(int)
    for ann in gt_annotations:
        issue = ann.get("issue")
        if issue:
            issue_stats[issue] += 1
    print(f"\n   Issues: Spurious {issue_stats['spurious']}, Location {issue_stats['location']}, "
          f"Label {issue_stats['label']}, Missing {len(missing_predictions)}")

    print(f"\n[4/5] Apply cleaning...")
    cleaned_annotations = []
    stats = {
        "deleted_spurious": 0, "added_missing": 0, "fixed_location": 0, "fixed_label": 0,
        "kept_normal": 0, "skipped_location": 0, "skipped_label": 0, "skipped_missing": 0
    }
    new_id_counter = 0

    for ann in gt_annotations:
        issue = ann.get("issue")

        if issue == "spurious":
            stats["deleted_spurious"] += 1
            continue

        if issue == "location":
            matching_pred = find_matching_prediction(ann, predictions_by_image)
            if matching_pred and matching_pred.get("score", 0) >= location_score_threshold:
                ann["bbox"] = matching_pred["bbox"]
                ann["area"] = matching_pred["bbox"][2] * matching_pred["bbox"][3]
                stats["fixed_location"] += 1
            else:
                stats["skipped_location"] += 1
                stats["kept_normal"] += 1
            for k in ("issue", "quality", "cluster", "category"):
                ann.pop(k, None)
            cleaned_annotations.append(ann)
            continue

        if issue == "label":
            matching_pred = find_matching_prediction(ann, predictions_by_image)
            if matching_pred and matching_pred.get("score", 0) >= label_score_threshold:
                ann["category_id"] = matching_pred["category_id"]
                stats["fixed_label"] += 1
            else:
                stats["skipped_label"] += 1
                stats["kept_normal"] += 1
            for k in ("issue", "quality", "cluster", "category"):
                ann.pop(k, None)
            cleaned_annotations.append(ann)
            continue

        for k in ("issue", "quality", "cluster", "category"):
            ann.pop(k, None)
        cleaned_annotations.append(ann)
        stats["kept_normal"] += 1

    print(f"\n[5/5] Add missing annotations...")
    for pred in missing_predictions:
        score = pred.get("score", 0)
        if score >= missing_score_threshold:
            new_ann = {
                "id": new_id_counter,
                "image_id": pred["image_id"],
                "category_id": pred["category_id"],
                "bbox": pred["bbox"],
                "area": pred["area"],
                "iscrowd": 0
            }
            cleaned_annotations.append(new_ann)
            stats["added_missing"] += 1
            new_id_counter += 1
        else:
            stats["skipped_missing"] += 1

    for i, ann in enumerate(cleaned_annotations):
        ann["id"] = i

    output_data = {
        "info": quality_report["info"],
        "licenses": quality_report["licenses"],
        "images": quality_report["images"],
        "annotations": cleaned_annotations,
        "categories": quality_report["categories"]
    }

    print(f"\n[Save] Write cleaned annotations to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("Cleaning done. Stats:")
    print("=" * 70)
    print(f"  Original GT: {len(gt_annotations)}")
    print(f"  Cleaned: {len(cleaned_annotations)}")
    print(f"\n  Deleted spurious: {stats['deleted_spurious']}")
    print(f"  Fixed location: {stats['fixed_location']}")
    print(f"  Fixed label: {stats['fixed_label']}")
    print(f"  Added missing: {stats['added_missing']}")
    print(f"  Kept normal: {stats['kept_normal']}")
    print(f"\n  Skipped location: {stats['skipped_location']}, label: {stats['skipped_label']}, missing: {stats['skipped_missing']}")
    print(f"  Net change: {len(cleaned_annotations) - len(gt_annotations)} annotations")
    print(f"\n  Output: {output_file}")
    print("=" * 70)
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-clean dataset (original unchanged)")
    parser.add_argument("--quality-report", type=str, default="quality_report.json", help="Quality report path")
    parser.add_argument("--predictions", type=str, default="predictions_coco.json", help="Predictions path")
    parser.add_argument("--output", type=str, default="cleaned_annotations.json", help="Output path")
    parser.add_argument("--location-threshold", type=float, default=0.7, help="Location fix score threshold (default: 0.7)")
    parser.add_argument("--label-threshold", type=float, default=0.8, help="Label fix score threshold (default: 0.8)")
    parser.add_argument("--missing-threshold", type=float, default=0.5, help="Missing add score threshold (default: 0.5)")
    args = parser.parse_args()
    clean_dataset(
        quality_report_file=args.quality_report,
        predictions_file=args.predictions,
        output_file=args.output,
        location_score_threshold=args.location_threshold,
        label_score_threshold=args.label_threshold,
        missing_score_threshold=args.missing_threshold
    )
