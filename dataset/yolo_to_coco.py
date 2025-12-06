#!/usr/bin/env python3
"""
将YOLO格式标注转换为COCO格式
用于SafeDNN-Clean分析

用法:
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
    转换YOLO格式到COCO格式
    
    Args:
        yolo_dir: YOLO数据集目录 (包含images和labels子目录)
        output_file: 输出的COCO格式JSON文件
        dataset_yaml: 数据集配置文件路径
    """
    images_dir = Path(yolo_dir) / "images"
    labels_dir = Path(yolo_dir) / "labels"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {labels_dir}")
    
    # 读取数据集配置
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    class_names = dataset_config['names']
    
    # COCO格式结构
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
    
    # 添加类别
    for class_id, class_name in class_names.items():
        coco_data["categories"].append({
            "id": int(class_id),
            "name": class_name,
            "supercategory": "weed"
        })
    
    # 处理每张图片
    image_id = 0
    annotation_id = 0
    
    image_files = sorted(images_dir.glob("*.jpg"))
    print(f"找到 {len(image_files)} 张图片")
    
    for img_path in image_files:
        try:
            # 读取图片尺寸
            img = Image.open(img_path)
            width, height = img.size
            
            # 添加图片信息
            coco_data["images"].append({
                "id": image_id,
                "file_name": img_path.name,
                "width": width,
                "height": height
            })
            
            # 读取对应的标签
            label_path = labels_dir / f"{img_path.stem}.txt"
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
                            
                            # 转换为COCO格式 (绝对坐标)
                            # YOLO格式: 归一化 [x_center, y_center, width, height]
                            # COCO格式: 绝对坐标 [x, y, width, height] (左上角)
                            x = (x_center - w/2) * width
                            y = (y_center - h/2) * height
                            w_px = w * width
                            h_px = h * height
                            
                            # 确保坐标在有效范围内
                            x = max(0, min(x, width - 1))
                            y = max(0, min(y, height - 1))
                            w_px = max(1, min(w_px, width - x))
                            h_px = max(1, min(h_px, height - y))
                            
                            # COCO格式: [x, y, width, height]
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
            print(f"警告: 处理图片 {img_path.name} 时出错: {e}")
            continue
        
        image_id += 1
    
    # 保存COCO格式文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 转换完成!")
    print(f"   图片数量: {len(coco_data['images'])}")
    print(f"   标注数量: {len(coco_data['annotations'])}")
    print(f"   类别数量: {len(coco_data['categories'])}")
    print(f"   保存到: {output_path.absolute()}")
    
    return coco_data


def main():
    parser = argparse.ArgumentParser(
        description="将YOLO格式标注转换为COCO格式"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="val",
        help="数据集分割 (train 或 val)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径 (默认: annotations_{split}_coco.json)"
    )
    parser.add_argument(
        "--dataset-yaml",
        type=str,
        default="dataset.yaml",
        help="数据集配置文件路径"
    )
    
    args = parser.parse_args()
    
    # 确定输出文件
    if args.output is None:
        args.output = f"annotations_{args.split}_coco.json"
    
    # 转换
    yolo_to_coco(args.split, args.output, args.dataset_yaml)


if __name__ == "__main__":
    main()

