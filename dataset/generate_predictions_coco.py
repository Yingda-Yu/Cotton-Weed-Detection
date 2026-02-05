#!/usr/bin/env python3
"""
Generate predictions on val/train and convert to COCO format for SafeDNN-Clean.

Usage:
    python generate_predictions_coco.py --model runs/detect/yolov8n_baseline/weights/best.pt \\
        --val-dir val --annotations annotations_val_coco.json --output predictions_coco.json
"""

import json
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np


def generate_predictions_coco(
    model_weights,
    val_dir_or_split="val",
    annotations_file="annotations_val_coco.json",
    output_file="predictions_coco.json",
    conf_threshold=0.25
):
    """
    Generate COCO-format predictions.

    Args:
        model_weights: Model weights path
        val_dir_or_split: Val/train dir or split name ("train" or "val")
        annotations_file: COCO annotations (for image_id mapping)
        output_file: Output path
        conf_threshold: Confidence threshold
    """
    model_path = Path(model_weights)
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if val_dir_or_split in ["train", "val"]:
        val_dir = f"cotton weed dataset/{val_dir_or_split}"
    else:
        val_dir = val_dir_or_split
    val_images_dir = Path(val_dir) / "images"
    if not val_images_dir.exists():
        raise FileNotFoundError(f"Val images dir not found: {val_images_dir}")
    annotations_path = Path(annotations_file)
    if not annotations_path.exists():
        raise FileNotFoundError(
            f"Annotations not found: {annotations_path}. Run: python yolo_to_coco.py --split val"
        )
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    with open("dataset.yaml", 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    class_names = dataset_config['names']
    print(f"Reading annotations: {annotations_path}")
    with open(annotations_path, 'r', encoding='utf-8') as f:
        coco_annotations = json.load(f)
    
    image_id_map = {}
    for img_info in coco_annotations["images"]:
        image_id_map[img_info["file_name"]] = img_info["id"]
    
    predictions = {
        "info": coco_annotations["info"],
        "licenses": coco_annotations["licenses"],
        "images": coco_annotations["images"],
        "annotations": [],
        "categories": coco_annotations["categories"]
    }
    
    image_files = sorted(val_images_dir.glob("*.jpg"))
    print(f"Val images: {len(image_files)}, in annotations: {len(image_id_map)}, conf: {conf_threshold}")
    print("Generating predictions...")
    
    annotation_id = 0
    total_predictions = 0
    skipped_count = 0
    
    for i, img_path in enumerate(image_files, 1):
        if img_path.name not in image_id_map:
            skipped_count += 1
            if skipped_count <= 20:
                print(f"Warning: {img_path.name} not in annotations, skipping.")
            elif skipped_count == 21:
                print("Warning: More images not in annotations, skipping silently...")
            continue
        try:
            results = model.predict(
                str(img_path),
                conf=conf_threshold,
                verbose=False,
                imgsz=640
            )
        except Exception as e:
            print(f"Warning: Error predicting {img_path.name}: {e}, skipping.")
            continue
        try:
            img = Image.open(img_path)
            width, height = img.size
        except Exception as e:
            print(f"Warning: Cannot read {img_path.name}: {e}, skipping.")
            continue
        if not results or len(results) == 0:
            continue
            
        for result in results:
            if not hasattr(result, 'boxes') or result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls)
                conf = float(box.conf)
                
                xywhn = box.xywhn[0].cpu().numpy()
                x_center, y_center, w, h = xywhn
                x = (x_center - w/2) * width
                y = (y_center - h/2) * height
                w_px = w * width
                h_px = h * height
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w_px = max(1, min(w_px, width - x))
                h_px = max(1, min(h_px, height - y))
                image_id = image_id_map[img_path.name]
                predictions["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "category": class_names[class_id],
                    "bbox": [float(x), float(y), float(w_px), float(h_px)],
                    "area": float(w_px * h_px),
                    "score": float(conf),
                    "iscrowd": 0
                })
                annotation_id += 1
                total_predictions += 1
        
        if i % 20 == 0:
            print(f"  Processed: {i}/{len(image_files)} images, {total_predictions} predictions")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    processed_count = len(image_files) - skipped_count
    print(f"\nDone. Images: {len(image_files)}, processed: {processed_count}, skipped: {skipped_count}, "
          f"predictions: {total_predictions}. Avg per image: {total_predictions/processed_count:.2f}" if processed_count else "")
    print(f"Saved: {output_path.absolute()}")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate COCO-format model predictions")
    parser.add_argument("--model", type=str, required=True, help="Model weights path")
    parser.add_argument("--val-dir", type=str, default="val", help="Val dir or split (train/val)")
    parser.add_argument("--annotations", type=str, default="annotations_val_coco.json", help="COCO annotations path")
    parser.add_argument("--output", type=str, default="predictions_coco.json", help="Output path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    
    args = parser.parse_args()
    
    generate_predictions_coco(
        args.model,
        args.val_dir,
        args.annotations,
        args.output,
        args.conf
    )


if __name__ == "__main__":
    main()

