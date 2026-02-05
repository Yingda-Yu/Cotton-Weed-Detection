#!/usr/bin/env python3
"""
Visualize training set annotations.
Sample images from the training set and draw YOLO-format labels on them.
"""

import cv2
import numpy as np
from pathlib import Path
import random
import yaml

NUM_SAMPLES = 10
OUTPUT_DIR = Path("visualized_samples")
TRAIN_IMAGES_DIR = Path("cotton weed dataset/train/images")
TRAIN_LABELS_DIR = Path("cotton weed dataset/train/labels")
DATASET_YAML = Path("dataset.yaml")

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
}

def load_dataset_config():
    """Load class names from dataset.yaml."""
    with open(DATASET_YAML, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['names']

# ============================================================================
# 读取和解析标签
# ============================================================================
def read_yolo_label(label_path):
    """Read YOLO label file. Returns [(class_id, x_center, y_center, width, height), ...]."""
    annotations = []
    if not label_path.exists():
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                annotations.append((class_id, x_center, y_center, width, height))
    
    return annotations

# ============================================================================
# 坐标转换
# ============================================================================
def yolo_to_pixel(bbox, img_width, img_height):
    """Convert YOLO normalized (x_center, y_center, w, h) to pixel (x1, y1, x2, y2)."""
    x_center, y_center, width, height = bbox
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)
    
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width - 1))
    y2 = max(0, min(y2, img_height - 1))
    
    return (x1, y1, x2, y2)

# ============================================================================
# 绘制标注
# ============================================================================
def draw_annotations(img, annotations, class_names):
    """Draw bounding boxes and class labels on image.
    """
    img_height, img_width = img.shape[:2]
    
    for class_id, x_center, y_center, width, height in annotations:
        x1, y1, x2, y2 = yolo_to_pixel((x_center, y_center, width, height), 
                                       img_width, img_height)
        
        class_name = class_names.get(class_id, f"Class_{class_id}")
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_id}:{class_name}"
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(img, 
                     (x1, y1 - text_height - baseline - 5),
                     (x1 + text_width, y1),
                     color, -1)
        
        cv2.putText(img, label,
                   (x1, y1 - baseline - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 255, 255), 2)
    
    return img

# ============================================================================
# 主函数
# ============================================================================
def main():
    print("=" * 70)
    print("Training data annotation visualization")
    print("=" * 70)
    
    # 读取类别名称
    class_names = load_dataset_config()
    print(f"\nClasses: {class_names}")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nOutput: {OUTPUT_DIR}")
    
    # 获取所有图像文件
    image_files = list(TRAIN_IMAGES_DIR.glob("*.jpg"))
    print(f"\nFound {len(image_files)} training images")
    
    if len(image_files) == 0:
        print("Error: No training images found!")
        return
    
    # 随机选择样本
    num_samples = min(NUM_SAMPLES, len(image_files))
    selected_images = random.sample(image_files, num_samples)
    print(f"\nSampling {num_samples} images")
    
    # 处理每张图像
    success_count = 0
    for i, img_path in enumerate(selected_images, 1):
        print(f"\n[{i}/{num_samples}] {img_path.name}")
        
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: Cannot read {img_path.name}")
            continue
        
        # 读取对应的标签文件
        label_path = TRAIN_LABELS_DIR / (img_path.stem + ".txt")
        annotations = read_yolo_label(label_path)
        
        if len(annotations) == 0:
            print(f"  No labels (negative sample)")
            annotated_img = img.copy()
            cv2.putText(annotated_img, "No annotations",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 255), 2)
        else:
            print(f"  {len(annotations)} annotations")
            annotated_img = draw_annotations(img.copy(), annotations, class_names)
        
        output_path = OUTPUT_DIR / f"annotated_{img_path.name}"
        cv2.imwrite(str(output_path), annotated_img)
        print(f"  Saved: {output_path}")
        success_count += 1
    print("\n" + "=" * 70)
    print(f"Done. Processed {success_count}/{num_samples} images. Output: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()

