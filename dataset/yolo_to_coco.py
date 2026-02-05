#!/usr/bin/env python3
"""
Convert YOLO-format annotations to COCO format for SafeDNN-Clean.

Usage:
    python yolo_to_coco.py --split train --output annotations_train_coco.json
    python yolo_to_coco.py --split val --output annotations_val_coco.json
"""

import json
import yaml
import argparse
from pathlib import Path
from PIL import Image


def yolo_to_coco(yolo_dir, output_file, dataset_yaml="dataset.yaml"):
    """
    Convert YOLO format to COCO format.

    Args:
        yolo_dir: Dataset dir with images/ and labels/ subdirs
        output_file: Output COCO JSON path
        dataset_yaml: Dataset config path
    """
    images_dir = Path(yolo_dir) / "images"
    labels_dir = Path(yolo_dir) / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    class_names = dataset_config['names']
    
    coco_data = {
        "info": {
            "description": "Cotton Weed Detection Dataset",
            "version": "1.0",
            "year": 2024
        },
        "licenses": [{
            "id": 1,
            "name": "Unknown",
            "url": ""
        }],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    for class_id, class_name in class_names.items():
        coco_data["categories"].append({
            "id": int(class_id),
            "name": class_name,
            "supercategory": "weed"
        })
    
    image_id = 0
    annotation_id = 0
    
    image_files = sorted(images_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} images.")
    skipped_images = 0
    for img_path in image_files:
        try:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                skipped_images += 1
                if skipped_images <= 10:
                    print(f"Warning: No label for {img_path.name}, skipping.")
                elif skipped_images == 11:
                    print("Warning: More images without labels, skipping silently...")
                continue
            img = Image.open(img_path)
            width, height = img.size
            
            coco_data["images"].append({
                "id": image_id,
                "file_name": img_path.name,
                "width": width,
                "height": height
            })
            
            if label_path.exists():
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            
                            # YOLO normalized -> COCO absolute [x, y, w, h]
                            x = (x_center - w/2) * width
                            y = (y_center - h/2) * height
                            w_px = w * width
                            h_px = h * height
                            
                            x = max(0, min(x, width - 1))
                            y = max(0, min(y, height - 1))
                            w_px = max(1, min(w_px, width - x))
                            h_px = max(1, min(h_px, height - y))
                            
                            coco_data["annotations"].append({
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": class_id,
                                "bbox": [x, y, w_px, h_px],
                                "area": w_px * h_px,
                                "iscrowd": 0
                            })
                            annotation_id += 1
        except Exception as e:
            print(f"Warning: Error processing {img_path.name}: {e}")
            continue
        
        image_id += 1
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    print(f"\nConversion done. Images: {len(image_files)}, with labels: {len(coco_data['images'])}, "
          f"skipped: {skipped_images}, annotations: {len(coco_data['annotations'])}, "
          f"categories: {len(coco_data['categories'])}. Saved: {output_path.absolute()}")
    
    return coco_data


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO annotations to COCO format")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="val", help="Split (train or val)")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: annotations_{split}_coco.json)")
    parser.add_argument("--dataset-yaml", type=str, default="dataset.yaml", help="Dataset config path")
    args = parser.parse_args()
    if args.output is None:
        args.output = f"annotations_{args.split}_coco.json"
    dataset_dir = f"cotton weed dataset/{args.split}"
    yolo_to_coco(dataset_dir, args.output, args.dataset_yaml)


if __name__ == "__main__":
    main()

