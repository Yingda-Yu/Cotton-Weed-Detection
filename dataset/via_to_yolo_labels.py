#!/usr/bin/env python3
"""
Convert VIA-format annotations to YOLO-format labels.

Usage:
    python dataset/via_to_yolo_labels.py
"""

import json
import yaml
from pathlib import Path
from PIL import Image
from collections import defaultdict

WORKSPACE_ROOT = Path(__file__).parent.parent
DATASET_ROOT = WORKSPACE_ROOT / "cotton weed dataset"

CLASS_NAME_TO_ID = {
    "carpetweed": 0,
    "morningglory": 1,
    "palmer_amaranth": 2
}


def via_to_yolo_label(via_file: Path, images_dir: Path, output_labels_dir: Path) -> bool:
    """Convert a single VIA JSON file to YOLO-format labels. Returns True on success."""
    try:
        with open(via_file, 'r', encoding='utf-8') as f:
            via_data = json.load(f)
        via_key = list(via_data.keys())[0]
        file_info = via_data[via_key]
        filename = file_info.get("filename", "")
        if not filename:
            return False
        base_name = Path(filename).stem
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            potential = images_dir / (base_name + ext)
            if potential.exists():
                img_path = potential
                break
        if img_path is None:
            print(f"Warning: Image not found: {filename}")
            return False
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"Warning: Cannot read image {img_path}: {e}")
            return False
        regions = file_info.get("regions", [])
        yolo_lines = []
        for region in regions:
            shape_attrs = region.get("shape_attributes", {})
            region_attrs = region.get("region_attributes", {})
            if shape_attrs.get("name") != "rect":
                continue
            x = shape_attrs.get("x", 0)
            y = shape_attrs.get("y", 0)
            w = shape_attrs.get("width", 0)
            h = shape_attrs.get("height", 0)
            class_name = region_attrs.get("class", "")
            class_id = CLASS_NAME_TO_ID.get(class_name.lower(), None)
            if class_id is None:
                print(f"Warning: Unknown class '{class_name}', skipping")
                continue
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        label_file = output_labels_dir / f"{base_name}.txt"
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        with open(label_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))
        return True
    except Exception as e:
        print(f"Error processing {via_file}: {e}")
        return False


def convert_split(split: str):
    """Convert annotations for the given split ('train' or 'val') to YOLO labels."""
    print("=" * 70)
    print(f"Converting {split} annotations to YOLO labels")
    print("=" * 70)
    annotations_dir = DATASET_ROOT / split / "annotations"
    images_dir = DATASET_ROOT / split / "images"
    labels_dir = DATASET_ROOT / split / "labels"
    if not annotations_dir.exists():
        print(f"Error: Annotations dir not found: {annotations_dir}")
        return
    if not images_dir.exists():
        print(f"Error: Images dir not found: {images_dir}")
        return
    labels_dir.mkdir(parents=True, exist_ok=True)
    via_files = list(annotations_dir.glob("*.json"))
    print(f"\nFound {len(via_files)} annotation files")
    converted = 0
    skipped = 0
    class_counts = defaultdict(int)
    for i, via_file in enumerate(via_files, 1):
        if via_to_yolo_label(via_file, images_dir, labels_dir):
            converted += 1
            try:
                with open(via_file, 'r', encoding='utf-8') as f:
                    via_data = json.load(f)
                    via_key = list(via_data.keys())[0]
                    regions = via_data[via_key].get("regions", [])
                    for region in regions:
                        class_name = region.get("region_attributes", {}).get("class", "")
                        if class_name:
                            class_counts[class_name.lower()] += 1
            except Exception:
                pass
        else:
            skipped += 1
        if i % 50 == 0:
            print(f"  Processed: {i}/{len(via_files)}")
    print("\n" + "=" * 70)
    print("Conversion done.")
    print(f"  Converted: {converted}, skipped: {skipped}, output: {labels_dir.absolute()}")
    if class_counts:
        print("  Class counts:")
        for class_name, count in sorted(class_counts.items()):
            print(f"    {class_name}: {count}")
    print("=" * 70)


def main():
    print("=" * 70)
    print("VIA annotations to YOLO labels")
    print("=" * 70)
    for split in ["train", "val"]:
        convert_split(split)
        print()
    print("All conversions done.")
    print(f"Dataset structure: {DATASET_ROOT / 'train' / 'labels'}, {DATASET_ROOT / 'val' / 'labels'}")


if __name__ == "__main__":
    main()
