#!/usr/bin/env python3
"""
可视化质量报告中的问题标注
生成问题图片的可视化，便于人工检查和修复

用法:
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
    """读取YOLO格式标签"""
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
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                boxes.append((class_id, x_center, y_center, w, h))
    return boxes


def yolo_to_pixel(bbox, img_width, img_height):
    """将YOLO格式转换为像素坐标"""
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
    可视化问题标注
    
    Args:
        quality_report_file: 质量报告文件
        val_dir: 验证集目录
        output_dir: 输出目录
        top_n: 每种问题类型显示前N个
    """
    # 读取质量报告
    with open(quality_report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    # 读取数据集配置
    with open("dataset.yaml", 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    class_names = dataset_config['names']
    
    # 类别颜色 (BGR格式)
    class_colors = {
        0: (0, 255, 0),      # 绿色 - carpetweed
        1: (255, 0, 0),      # 蓝色 - morningglory
        2: (0, 0, 255),      # 红色 - palmer_amaranth
    }
    
    # 问题类型颜色
    issue_colors = {
        "spurious": (0, 0, 255),    # 红色
        "missing": (255, 0, 0),     # 蓝色
        "location": (0, 255, 255),   # 黄色
        "label": (0, 165, 255),     # 橙色
    }
    
    # 按问题类型分组
    issues_by_type = defaultdict(list)
    for ann in report["annotations"]:
        if "issue" in ann:
            issues_by_type[ann["issue"]].append(ann)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 处理路径：如果是"train"或"val"，转换为新路径
    if val_dir in ["train", "val"]:
        val_dir = f"cotton weed dataset/{val_dir}"
    
    val_images_dir = Path(val_dir) / "images"
    val_labels_dir = Path(val_dir) / "labels"
    
    # 创建image_id到文件名的映射
    image_id_map = {}
    for img_info in report["images"]:
        image_id_map[img_info["id"]] = img_info["file_name"]
    
    print("=" * 70)
    print("可视化问题标注")
    print("=" * 70)
    
    # 处理每种问题类型
    for issue_type, annotations in issues_by_type.items():
        # 按质量分数排序（质量越低越严重）
        annotations.sort(key=lambda x: x.get("quality", 1.0))
        
        print(f"\n{issue_type}: {len(annotations)} 个问题")
        print(f"  可视化前 {min(top_n, len(annotations))} 个最严重的问题...")
        
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
            
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
            
            # 读取真实标注
            label_path = val_labels_dir / f"{Path(image_name).stem}.txt"
            gt_boxes = read_yolo_label(label_path)
            
            # 绘制真实标注（绿色）
            for gt_box in gt_boxes:
                class_id, (x1, y1, x2, y2) = yolo_to_pixel(gt_box, img_width, img_height)
                color = class_colors.get(class_id, (255, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # 标注类别
                class_name = class_names.get(class_id, f"Class_{class_id}")
                label = f"GT:{class_name}"
                cv2.putText(img, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 如果是spurious类型，标注有问题的地方
            if issue_type == "spurious":
                # 从COCO格式的bbox绘制
                if "bbox" in ann:
                    x, y, w, h = ann["bbox"]
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)
                    issue_color = issue_colors[issue_type]
                    cv2.rectangle(img, (x1, y1), (x2, y2), issue_color, 3)
                    cv2.putText(img, f"SPURIOUS (q:{ann.get('quality', 0):.3f})",
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                               issue_color, 2)
            
            # 添加问题信息
            quality = ann.get("quality", 0)
            info_text = f"{issue_type.upper()} - Quality: {quality:.3f}"
            cv2.putText(img, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, issue_colors[issue_type], 2)
            
            # 保存
            output_file = issue_output_dir / f"{Path(image_name).stem}_q{quality:.3f}.jpg"
            cv2.imwrite(str(output_file), img)
            count += 1
        
        print(f"  ✅ 已保存 {count} 张可视化图片到: {issue_output_dir}")
    
    print("\n" + "=" * 70)
    print(f"✅ 可视化完成!")
    print(f"   输出目录: {output_path.absolute()}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="可视化质量报告中的问题标注"
    )
    parser.add_argument(
        "--report",
        type=str,
        default="quality_report.json",
        help="质量报告文件"
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="cotton weed dataset/val",
        help="验证集目录或split名称（train/val）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="quality_issues",
        help="输出目录"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="每种问题类型显示前N个 (默认: 50)"
    )
    
    args = parser.parse_args()
    
    visualize_issues(
        args.report,
        args.val_dir,
        args.output,
        args.top_n
    )


if __name__ == "__main__":
    main()

