#!/usr/bin/env python3
"""
Visualize problematic annotations from the quality report.
Generates visualizations for manual inspection and fixing.

Usage:
    python visualize_quality_report.py \
        --report quality_report.json \
        --val-dir val \
        --output quality_issues
"""

import json
import cv2
import argparse
from pathlib import Path
from collections import defaultdict
import yaml


def read_yolo_label(label_path):
    """Read YOLO-format label file."""
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center, y_center = float(parts[1]), float(parts[2])
                w, h = float(parts[3]), float(parts[4])
                boxes.append((class_id, x_center, y_center, w, h))
    return boxes


def yolo_to_pixel(bbox, img_width, img_height):
    """Convert YOLO normalized coords to pixel coords."""
    class_id, x_center, y_center, w, h = bbox
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = w * img_width
    height_px = h * img_height
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width - 1))
    y2 = max(0, min(y2, img_height - 1))
    return class_id, (x1, y1, x2, y2)


def visualize_issues(
    quality_report_file="quality_report.json",
    val_dir="cotton weed dataset/val",
    output_dir="quality_issues",
    top_n=50
):
    """
    Visualize problematic annotations from the quality report.

    Args:
        quality_report_file: Path to quality report JSON
        val_dir: Validation (or train) directory
        output_dir: Output directory for images
        top_n: Max number of samples per issue type to visualize
    """
    with open(quality_report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)
    with open("dataset.yaml", 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    class_names = dataset_config['names']

    class_colors = {
        0: (0, 255, 0),    # green - carpetweed
        1: (255, 0, 0),    # blue - morningglory
        2: (0, 0, 255),    # red - palmer_amaranth
    }
    issue_colors = {
        "spurious": (0, 0, 255),
        "missing": (255, 0, 0),
        "location": (0, 255, 255),
        "label": (0, 165, 255),
    }

    issues_by_type = defaultdict(list)
    for ann in report["annotations"]:
        if "issue" in ann:
            issues_by_type[ann["issue"]].append(ann)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if val_dir in ["train", "val"]:
        val_dir = f"cotton weed dataset/{val_dir}"
    val_images_dir = Path(val_dir) / "images"
    val_labels_dir = Path(val_dir) / "labels"

    image_id_map = {img_info["id"]: img_info["file_name"] for img_info in report["images"]}

    print("=" * 70)
    print("Visualize problematic annotations")
    print("=" * 70)

    for issue_type, annotations in issues_by_type.items():
        annotations.sort(key=lambda x: x.get("quality", 1.0))
        n_show = min(top_n, len(annotations))
        print(f"\n{issue_type}: {len(annotations)} issues, showing top {n_show}")

        issue_output_dir = output_path / issue_type
        issue_output_dir.mkdir(exist_ok=True)
        count = 0

        for ann in annotations[:top_n]:
            image_id = ann["image_id"]
            image_name = image_id_map.get(image_id)
            if not image_name:
                continue
            img_path = val_images_dir / image_name
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_height, img_width = img.shape[:2]
            label_path = val_labels_dir / f"{Path(image_name).stem}.txt"
            gt_boxes = read_yolo_label(label_path)

            for gt_box in gt_boxes:
                class_id, (x1, y1, x2, y2) = yolo_to_pixel(gt_box, img_width, img_height)
                color = class_colors.get(class_id, (255, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                class_name = class_names.get(class_id, f"Class_{class_id}")
                cv2.putText(img, f"GT:{class_name}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if issue_type == "spurious" and "bbox" in ann:
                x, y, w, h = ann["bbox"]
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                issue_color = issue_colors[issue_type]
                cv2.rectangle(img, (x1, y1), (x2, y2), issue_color, 3)
                cv2.putText(img, f"SPURIOUS (q:{ann.get('quality', 0):.3f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, issue_color, 2)

            quality = ann.get("quality", 0)
            info_text = f"{issue_type.upper()} - Quality: {quality:.3f}"
            cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, issue_colors[issue_type], 2)
            output_file = issue_output_dir / f"{Path(image_name).stem}_q{quality:.3f}.jpg"
            cv2.imwrite(str(output_file), img)
            count += 1

        print(f"  Saved {count} images to {issue_output_dir}")

    print("\n" + "=" * 70)
    print(f"Done. Output: {output_path.absolute()}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Visualize problematic annotations from quality report")
    parser.add_argument("--report", type=str, default="quality_report.json", help="Quality report path")
    parser.add_argument("--val-dir", type=str, default="cotton weed dataset/val", help="Val/train directory or split name")
    parser.add_argument("--output", type=str, default="quality_issues", help="Output directory")
    parser.add_argument("--top-n", type=int, default=50, help="Max samples per issue type (default: 50)")
    args = parser.parse_args()
    visualize_issues(args.report, args.val_dir, args.output, args.top_n)


if __name__ == "__main__":
    main()
