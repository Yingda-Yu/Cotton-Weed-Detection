#!/usr/bin/env python3
"""
Visualize problematic annotations from the quality report.
1. Show top-N lowest-quality (least reliable) annotations
2. Show examples by issue type (spurious, location, label, missing).

Usage:
    python tools/visualize_quality_issues.py \
        --quality-report quality_report_train.json \
        --annotations annotations_train_coco.json \
        --images-dir "cotton weed dataset/train/images" \
        --output-dir quality_issues_visualization \
        --top-n 20
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

COLORS = {
    "spurious": (255, 0, 0),
    "location": (255, 165, 0),
    "label": (255, 0, 255),
    "missing": (255, 255, 0),
    "normal": (0, 255, 0),
}

CATEGORY_NAMES = {
    0: "Carpetweed",
    1: "Morning Glory",
    2: "Palmer Amaranth"
}


def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def bbox_to_xyxy(bbox):
    """Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def draw_bbox_with_label(draw, bbox, color, label, width=5):
    """Draw bbox and label."""
    x1, y1, x2, y2 = bbox_to_xyxy(bbox)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 40)
        except:
            font = ImageFont.load_default()
    
    bbox_text = draw.textbbox((0, 0), label, font=font)
    text_width = bbox_text[2] - bbox_text[0]
    text_height = bbox_text[3] - bbox_text[1]
    label_y = max(0, y1 - text_height - 10)
    
    draw.rectangle(
        [x1, label_y, x1 + text_width + 20, label_y + text_height + 10],
        fill=(0, 0, 0, 240)
    )
    
    draw.text((x1 + 10, label_y + 5), label, fill=(255, 255, 255), font=font)


def visualize_lowest_quality(
    quality_report,
    annotations_data,
    images_dir,
    output_dir,
    top_n=20
):
    """Visualize top-N lowest-quality annotations."""
    print("\n" + "=" * 70)
    print(f"Top {top_n} lowest-quality annotations")
    print("=" * 70)
    
    gt_annotations = []
    for ann in quality_report["annotations"]:
        if ann.get("id", 0) >= 0:
            quality = ann.get("quality", 1.0)
            issue = ann.get("issue", None)
            gt_annotations.append({
                "ann": ann,
                "quality": quality,
                "issue": issue
            })
    
    gt_annotations.sort(key=lambda x: x["quality"])
    
    top_annotations = gt_annotations[:top_n]
    
    print(f"Found {len(top_annotations)} lowest-quality annotations")
    
    output_path = Path(output_dir) / "lowest_quality"
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_id_to_info = {img['id']: img for img in annotations_data['images']}
    
    annotations_by_image = defaultdict(list)
    for item in top_annotations:
        ann = item["ann"]
        image_id = ann["image_id"]
        annotations_by_image[image_id].append(item)
    
    success_count = 0
    for image_id, items in annotations_by_image.items():
        image_info = image_id_to_info.get(image_id)
        if not image_info:
            continue
        
        image_filename = image_info['file_name']
        image_path = Path(images_dir) / image_filename
        
        if not image_path.exists():
            continue
        
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        for item in items:
            ann = item["ann"]
            quality = item["quality"]
            issue = item["issue"]
            category_id = ann['category_id']
            category_name = CATEGORY_NAMES.get(category_id, f"Class {category_id}")
            
            if issue == "spurious":
                color = COLORS["spurious"]
                label = f"{category_name} [SPURIOUS] Quality: {quality:.4f}"
            elif issue == "location":
                color = COLORS["location"]
                label = f"{category_name} [LOCATION] Quality: {quality:.4f}"
            elif issue == "label":
                color = COLORS["label"]
                label = f"{category_name} [LABEL] Quality: {quality:.4f}"
            else:
                color = COLORS["normal"]
                label = f"{category_name} Quality: {quality:.4f}"
            
            draw_bbox_with_label(draw, ann['bbox'], color, label, width=6)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 50)
        except:
            try:
                title_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 50)
            except:
                title_font = ImageFont.load_default()
        
        title = f"Lowest Quality Annotations (Top {len(items)})"
        draw.text((10, 10), title, fill=(255, 255, 255), font=title_font, 
                 stroke_width=3, stroke_fill=(0, 0, 0))
        
        output_filename = f"{image_id:05d}_{image_filename}"
        output_file = output_path / output_filename
        img.save(output_file, quality=95)
        success_count += 1
    
    print(f"Generated {success_count} images to: {output_path}")
    return success_count


def visualize_by_issue_type(
    quality_report,
    annotations_data,
    images_dir,
    output_dir,
    samples_per_type=10
):
    """Visualize examples by issue type."""
    print("\n" + "=" * 70)
    print(f"By issue type ({samples_per_type} per type)")
    print("=" * 70)
    
    issues_by_type = {
        "spurious": [],
        "location": [],
        "label": [],
        "missing": []
    }
    
    for ann in quality_report["annotations"]:
        if ann.get("id", 0) >= 0:
            issue = ann.get("issue")
            if issue in issues_by_type:
                quality = ann.get("quality", 1.0)
                issues_by_type[issue].append({
                    "ann": ann,
                    "quality": quality
                })
    
    for ann in quality_report["annotations"]:
        if ann.get("id", 0) < 0:
            issues_by_type["missing"].append({
                "ann": ann,
                "quality": ann.get("quality", 0.0)
            })
    
    for issue_type in issues_by_type:
        issues_by_type[issue_type].sort(key=lambda x: x["quality"])
        print(f"  {issue_type}: {len(issues_by_type[issue_type])} issues")
    image_id_to_info = {img['id']: img for img in annotations_data['images']}
    
    for issue_type, items in issues_by_type.items():
        if len(items) == 0:
            print(f"\n{issue_type}: no issues found")
            continue
        
        print(f"\nProcessing {issue_type}...")
        
        output_path = Path(output_dir) / issue_type
        output_path.mkdir(parents=True, exist_ok=True)
        
        samples = items[:samples_per_type]
        
        annotations_by_image = defaultdict(list)
        for item in samples:
            ann = item["ann"]
            image_id = ann["image_id"]
            annotations_by_image[image_id].append(item)
        
        success_count = 0
        for image_id, image_items in annotations_by_image.items():
            image_info = image_id_to_info.get(image_id)
            if not image_info:
                continue
            
            image_filename = image_info['file_name']
            image_path = Path(images_dir) / image_filename
            
            if not image_path.exists():
                continue
            
            img = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(img)
            
            for item in image_items:
                ann = item["ann"]
                quality = item["quality"]
                category_id = ann.get('category_id', -1)
                
                if issue_type == "missing":
                    if 'category' in ann:
                        category_name = ann['category']
                    else:
                        category_name = f"Class {category_id}"
                    label = f"{category_name} [MISSING] Quality: {quality:.4f}"
                else:
                    category_name = CATEGORY_NAMES.get(category_id, f"Class {category_id}")
                    label = f"{category_name} [{issue_type.upper()}] Quality: {quality:.4f}"
                
                color = COLORS[issue_type]
                draw_bbox_with_label(draw, ann['bbox'], color, label, width=6)
            
            try:
                title_font = ImageFont.truetype("arial.ttf", 50)
            except:
                try:
                    title_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 50)
                except:
                    title_font = ImageFont.load_default()
            
            title = f"{issue_type.upper()} Issue Example"
            draw.text((10, 10), title, fill=(255, 255, 255), font=title_font,
                     stroke_width=3, stroke_fill=(0, 0, 0))
            
            output_filename = f"{image_id:05d}_{image_filename}"
            output_file = output_path / output_filename
            img.save(output_file, quality=95)
            success_count += 1
        
        print(f"  Generated {success_count} images to: {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Visualize problematic annotations from quality report"
    )
    parser.add_argument("--quality-report", type=str, default="quality_report_train.json", help="Quality report path")
    parser.add_argument("--annotations", type=str, default="annotations_train_coco.json", help="COCO annotations path")
    parser.add_argument("--images-dir", type=str, default="cotton weed dataset/train/images", help="Images directory")
    parser.add_argument("--output-dir", type=str, default="quality_issues_visualization", help="Output directory")
    parser.add_argument("--top-n", type=int, default=20, help="Top-N lowest quality (default: 20)")
    parser.add_argument("--samples-per-type", type=int, default=10, help="Samples per issue type (default: 10)")
    args = parser.parse_args()
    if not Path(args.quality_report).exists():
        print(f"ERROR: Quality report file not found: {args.quality_report}")
        return 1
    
    if not Path(args.annotations).exists():
        print(f"ERROR: Annotations file not found: {args.annotations}")
        return 1
    
    print("=" * 70)
    print("Quality report visualization")
    print("=" * 70)
    print(f"Report: {args.quality_report}, Annotations: {args.annotations}, Images: {args.images_dir}, Output: {args.output_dir}")
    print("=" * 70)
    print("\nLoading data...")
    quality_report = load_json(args.quality_report)
    annotations_data = load_json(args.annotations)
    
    print(f"  Report annotations: {len(quality_report['annotations'])}, Images: {len(annotations_data['images'])}")
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    visualize_lowest_quality(
        quality_report,
        annotations_data,
        args.images_dir,
        args.output_dir,
        args.top_n
    )
    
    visualize_by_issue_type(
        quality_report,
        annotations_data,
        args.images_dir,
        args.output_dir,
        args.samples_per_type
    )
    
    print("\n" + "=" * 70)
    print("✅ 可视化完成！")
    print("=" * 70)
    print(f"Output: {output_path.absolute()}")
    print("Folders: lowest_quality/, spurious/, location/, label/, missing/")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())

