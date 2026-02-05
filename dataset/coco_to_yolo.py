#!/usr/bin/env python3
"""
Convert COCO-format annotations back to YOLO format.
Used to write cleaned annotations to a YOLO dataset. Original data is not overwritten.
Default output: cleaned_{split}/labels/

Usage:
    python coco_to_yolo.py --coco-file cleaned_annotations.json --split val --output-dir cleaned_val
"""

import json
import yaml
import argparse
from pathlib import Path
from PIL import Image


def coco_to_yolo(
    coco_file,
    split_dir,
    output_dir=None,
    images_dir=None,
    dataset_yaml="dataset.yaml"
):
    """
    Convert COCO format to YOLO format.

    Args:
        coco_file: COCO JSON path
        split_dir: Original split dir (for image paths/sizes)
        output_dir: Output dir (default: cleaned_{split_dir})
        images_dir: Images dir (default: {split_dir}/images)
        dataset_yaml: Dataset config path
    """
    coco_path = Path(coco_file)
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO file not found: {coco_path}")
    split_path = Path(split_dir)
    if not split_path.exists():
        raise FileNotFoundError(f"Split dir not found: {split_path}")
    if output_dir is None:
        output_dir = f"cleaned_{split_dir}"
    output_path = Path(output_dir)
    if images_dir is None:
        images_dir = split_path / "images"
    else:
        images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    output_labels_dir = output_path / "labels"
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    class_names = dataset_config['names']
    
    print(f"Reading COCO file: {coco_path}")
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    image_id_to_filename = {}
    image_id_to_size = {}
    for img_info in coco_data["images"]:
        image_id_to_filename[img_info["id"]] = img_info["file_name"]
        image_id_to_size[img_info["id"]] = (img_info["width"], img_info["height"])
    
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    print(f"Found annotations for {len(annotations_by_image)} images.")
    converted_count = 0
    skipped_count = 0
    total_annotations = 0
    
    for image_id, annotations in annotations_by_image.items():
        filename = image_id_to_filename.get(image_id)
        if not filename:
            print(f"Warning: No filename for image_id={image_id}, skipping.")
            skipped_count += 1
            continue
        width, height = image_id_to_size.get(image_id, (None, None))
        if width is None or height is None:
            img_path = images_dir / filename
            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    width, height = img.size
                except Exception as e:
                    print(f"Warning: Cannot read size of {filename}: {e}, skipping.")
                    skipped_count += 1
                    continue
            else:
                print(f"Warning: Image not found {filename}, skipping.")
                skipped_count += 1
                continue
        label_filename = Path(filename).stem + ".txt"
        label_path = output_labels_dir / label_filename
        
        yolo_lines = []
        for ann in annotations:
            category_id = ann["category_id"]
            bbox = ann["bbox"]
            x, y, w, h = bbox
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height
            
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))
            
            yolo_lines.append(f"{category_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))
        
        converted_count += 1
        total_annotations += len(annotations)
    
    print("\n" + "=" * 70)
    print("Conversion done.")
    print(f"  Converted: {converted_count}, skipped: {skipped_count}, annotations: {total_annotations}")
    print(f"  Output: {output_labels_dir.absolute()}. Original data unchanged.")
    print("=" * 70)
    
    return output_labels_dir


def main():
    parser = argparse.ArgumentParser(description="Convert COCO to YOLO format (original data unchanged)")
    parser.add_argument("--coco-file", type=str, required=True, help="COCO JSON path")
    parser.add_argument("--split", type=str, default="val", help="Split dir (e.g. val or train)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output dir (default: cleaned_{split})")
    parser.add_argument("--images-dir", type=str, default=None, help="Images dir (default: {split}/images)")
    parser.add_argument("--dataset-yaml", type=str, default="dataset.yaml", help="Dataset config path")
    
    args = parser.parse_args()
    
    coco_to_yolo(
        coco_file=args.coco_file,
        split_dir=args.split,
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        dataset_yaml=args.dataset_yaml
    )


if __name__ == "__main__":
    main()

