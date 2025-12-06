#!/usr/bin/env python3
"""
可视化训练数据中的标注
从训练集中抽取样本，将YOLO格式的标签绘制在图像上
"""

import cv2
import numpy as np
from pathlib import Path
import random
import yaml

# ============================================================================
# 配置
# ============================================================================
NUM_SAMPLES = 10  # 要可视化的样本数量
OUTPUT_DIR = Path("visualized_samples")  # 输出目录
TRAIN_IMAGES_DIR = Path("train/images")
TRAIN_LABELS_DIR = Path("train/labels")
DATASET_YAML = Path("dataset.yaml")

# 类别颜色 (BGR格式，OpenCV使用)
CLASS_COLORS = {
    0: (0, 255, 0),      # 绿色 - carpetweed
    1: (255, 0, 0),      # 蓝色 - morningglory
    2: (0, 0, 255),      # 红色 - palmer_amaranth
}

# ============================================================================
# 读取数据集配置
# ============================================================================
def load_dataset_config():
    """读取dataset.yaml获取类别名称"""
    with open(DATASET_YAML, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['names']

# ============================================================================
# 读取和解析标签
# ============================================================================
def read_yolo_label(label_path):
    """
    读取YOLO格式的标签文件
    返回: [(class_id, x_center, y_center, width, height), ...]
    """
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
    """
    将YOLO格式的归一化坐标转换为像素坐标
    YOLO格式: (x_center, y_center, width, height) - 归一化到[0,1]
    返回: (x1, y1, x2, y2) - 像素坐标
    """
    x_center, y_center, width, height = bbox
    
    # 转换为像素坐标
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # 计算边界框的左上角和右下角
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)
    
    # 确保坐标在图像范围内
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width - 1))
    y2 = max(0, min(y2, img_height - 1))
    
    return (x1, y1, x2, y2)

# ============================================================================
# 绘制标注
# ============================================================================
def draw_annotations(img, annotations, class_names):
    """
    在图像上绘制边界框和类别标签
    """
    img_height, img_width = img.shape[:2]
    
    for class_id, x_center, y_center, width, height in annotations:
        # 转换坐标
        x1, y1, x2, y2 = yolo_to_pixel((x_center, y_center, width, height), 
                                       img_width, img_height)
        
        # 获取类别名称和颜色
        class_name = class_names.get(class_id, f"Class_{class_id}")
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        
        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本
        label = f"{class_id}:{class_name}"
        
        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # 绘制文本背景
        cv2.rectangle(img, 
                     (x1, y1 - text_height - baseline - 5),
                     (x1 + text_width, y1),
                     color, -1)
        
        # 绘制文本
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
    print("训练数据标注可视化")
    print("=" * 70)
    
    # 读取类别名称
    class_names = load_dataset_config()
    print(f"\n类别: {class_names}")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\n输出目录: {OUTPUT_DIR}")
    
    # 获取所有图像文件
    image_files = list(TRAIN_IMAGES_DIR.glob("*.jpg"))
    print(f"\n找到 {len(image_files)} 张训练图像")
    
    if len(image_files) == 0:
        print("错误: 未找到训练图像!")
        return
    
    # 随机选择样本
    num_samples = min(NUM_SAMPLES, len(image_files))
    selected_images = random.sample(image_files, num_samples)
    print(f"\n随机选择 {num_samples} 张图像进行可视化")
    
    # 处理每张图像
    success_count = 0
    for i, img_path in enumerate(selected_images, 1):
        print(f"\n[{i}/{num_samples}] 处理: {img_path.name}")
        
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  警告: 无法读取图像 {img_path.name}")
            continue
        
        # 读取对应的标签文件
        label_path = TRAIN_LABELS_DIR / (img_path.stem + ".txt")
        annotations = read_yolo_label(label_path)
        
        if len(annotations) == 0:
            print(f"  信息: 该图像没有标注 (可能是负样本)")
            # 仍然保存图像，标注"无标注"
            annotated_img = img.copy()
            cv2.putText(annotated_img, "No annotations",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 255), 2)
        else:
            print(f"  找到 {len(annotations)} 个标注")
            # 绘制标注
            annotated_img = draw_annotations(img.copy(), annotations, class_names)
        
        # 保存结果
        output_path = OUTPUT_DIR / f"annotated_{img_path.name}"
        cv2.imwrite(str(output_path), annotated_img)
        print(f"  已保存: {output_path}")
        success_count += 1
    
    print("\n" + "=" * 70)
    print(f"完成! 成功处理 {success_count}/{num_samples} 张图像")
    print(f"可视化结果保存在: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()

