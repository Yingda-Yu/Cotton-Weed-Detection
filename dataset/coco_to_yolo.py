#!/usr/bin/env python3
"""
将COCO格式标注转换回YOLO格式
用于将清洗后的标注写回YOLO格式数据集

⚠️ 注意：此脚本会创建新的YOLO格式标注文件，不会覆盖原始数据
默认输出到 cleaned_{split}/labels/ 目录

用法:
    python coco_to_yolo.py \
        --coco-file cleaned_annotations.json \
        --split val \
        --output-dir cleaned_val
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
    将COCO格式标注转换为YOLO格式
    
    Args:
        coco_file: COCO格式的JSON文件路径
        split_dir: 原始数据集目录（用于获取图片路径和尺寸）
        output_dir: 输出目录（默认: cleaned_{split_dir}）
        images_dir: 图片目录（默认: {split_dir}/images）
        dataset_yaml: 数据集配置文件路径
    """
    coco_path = Path(coco_file)
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO文件不存在: {coco_path}")
    
    split_path = Path(split_dir)
    if not split_path.exists():
        raise FileNotFoundError(f"数据集目录不存在: {split_path}")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = f"cleaned_{split_dir}"
    output_path = Path(output_dir)
    
    # 确定图片目录
    if images_dir is None:
        images_dir = split_path / "images"
    else:
        images_dir = Path(images_dir)
    
    if not images_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")
    
    # 创建输出目录
    output_labels_dir = output_path / "labels"
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取数据集配置
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    class_names = dataset_config['names']
    
    # 读取COCO格式文件
    print(f"读取COCO文件: {coco_path}")
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 创建image_id到文件名的映射
    image_id_to_filename = {}
    image_id_to_size = {}
    for img_info in coco_data["images"]:
        image_id_to_filename[img_info["id"]] = img_info["file_name"]
        image_id_to_size[img_info["id"]] = (img_info["width"], img_info["height"])
    
    # 按image_id组织标注
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    print(f"找到 {len(annotations_by_image)} 张图片的标注")
    
    # 转换每张图片的标注
    converted_count = 0
    skipped_count = 0
    total_annotations = 0
    
    for image_id, annotations in annotations_by_image.items():
        filename = image_id_to_filename.get(image_id)
        if not filename:
            print(f"警告: 找不到image_id={image_id}对应的文件名，跳过")
            skipped_count += 1
            continue
        
        # 获取图片尺寸
        width, height = image_id_to_size.get(image_id, (None, None))
        if width is None or height is None:
            # 尝试从图片文件读取
            img_path = images_dir / filename
            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    width, height = img.size
                except Exception as e:
                    print(f"警告: 无法读取图片 {filename} 的尺寸: {e}，跳过")
                    skipped_count += 1
                    continue
            else:
                print(f"警告: 找不到图片文件 {filename}，跳过")
                skipped_count += 1
                continue
        
        # 创建YOLO格式的标注文件
        label_filename = Path(filename).stem + ".txt"
        label_path = output_labels_dir / label_filename
        
        yolo_lines = []
        for ann in annotations:
            category_id = ann["category_id"]
            bbox = ann["bbox"]  # COCO格式: [x, y, width, height]
            
            x, y, w, h = bbox
            
            # 转换为YOLO格式: 归一化 [x_center, y_center, width, height]
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height
            
            # 确保值在[0, 1]范围内
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))
            
            # YOLO格式: class_id x_center y_center width height
            yolo_lines.append(f"{category_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # 写入YOLO格式文件
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))
        
        converted_count += 1
        total_annotations += len(annotations)
    
    print("\n" + "=" * 70)
    print("转换完成！")
    print("=" * 70)
    print(f"  转换图片数: {converted_count}")
    print(f"  跳过图片数: {skipped_count}")
    print(f"  总标注数: {total_annotations}")
    print(f"  输出目录: {output_labels_dir.absolute()}")
    print(f"\n  ✅ 原始数据集未修改，清洗后的标注已保存到: {output_dir}")
    print("=" * 70)
    
    return output_labels_dir


def main():
    parser = argparse.ArgumentParser(
        description="将COCO格式标注转换回YOLO格式（保留原始数据）"
    )
    parser.add_argument(
        "--coco-file",
        type=str,
        required=True,
        help="COCO格式的JSON文件路径（如 cleaned_annotations.json）"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="原始数据集目录（如 'val' 或 'train'）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认: cleaned_{split}）"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="图片目录（默认: {split}/images）"
    )
    parser.add_argument(
        "--dataset-yaml",
        type=str,
        default="dataset.yaml",
        help="数据集配置文件路径"
    )
    
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

