#!/usr/bin/env python3
"""
在验证集上生成预测，并转换为COCO格式
用于SafeDNN-Clean分析

用法:
    python generate_predictions_coco.py \
        --model runs/detect/yolov8n_baseline/weights/best.pt \
        --val-dir val \
        --annotations annotations_val_coco.json \
        --output predictions_coco.json
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
    生成COCO格式的预测结果
    
    Args:
        model_weights: 模型权重路径
        val_dir_or_split: 验证集目录或split名称（"train"或"val"）
        annotations_file: COCO格式的标注文件（用于获取image_id映射）
        output_file: 输出文件
        conf_threshold: 置信度阈值
    """
    # 检查文件
    model_path = Path(model_weights)
    if not model_path.exists():
        raise FileNotFoundError(f"模型权重不存在: {model_path}")
    
    # 处理路径：如果是"train"或"val"，转换为新路径
    if val_dir_or_split in ["train", "val"]:
        val_dir = f"cotton weed dataset/{val_dir_or_split}"
    else:
        val_dir = val_dir_or_split
    
    val_images_dir = Path(val_dir) / "images"
    if not val_images_dir.exists():
        raise FileNotFoundError(f"验证集图片目录不存在: {val_images_dir}")
    
    annotations_path = Path(annotations_file)
    if not annotations_path.exists():
        raise FileNotFoundError(
            f"标注文件不存在: {annotations_path}\n"
            f"请先运行: python yolo_to_coco.py --split val"
        )
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = YOLO(str(model_path))
    
    # 读取数据集配置
    with open("dataset.yaml", 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    class_names = dataset_config['names']
    
    # 读取COCO格式的标注（用于获取image_id映射和结构）
    print(f"读取标注文件: {annotations_path}")
    with open(annotations_path, 'r', encoding='utf-8') as f:
        coco_annotations = json.load(f)
    
    # 创建image_id映射 (file_name -> image_id)
    image_id_map = {}
    for img_info in coco_annotations["images"]:
        image_id_map[img_info["file_name"]] = img_info["id"]
    
    # COCO格式的预测结果（复用标注文件的结构）
    predictions = {
        "info": coco_annotations["info"],
        "licenses": coco_annotations["licenses"],
        "images": coco_annotations["images"],
        "annotations": [],
        "categories": coco_annotations["categories"]
    }
    
    # 获取所有图片文件
    image_files = sorted(val_images_dir.glob("*.jpg"))
    print(f"找到 {len(image_files)} 张验证图片")
    print(f"标注文件中有 {len(image_id_map)} 张图片")
    print(f"置信度阈值: {conf_threshold}")
    print("\n正在生成预测...")
    
    annotation_id = 0
    total_predictions = 0
    skipped_count = 0
    
    for i, img_path in enumerate(image_files, 1):
        if img_path.name not in image_id_map:
            skipped_count += 1
            if skipped_count <= 20:  # 只显示前20个警告
                print(f"警告: 图片 {img_path.name} 不在标注文件中，跳过")
            elif skipped_count == 21:
                print(f"警告: 还有更多图片不在标注文件中，将静默跳过...")
            continue
        
        # 运行预测
        try:
            results = model.predict(
                str(img_path),
                conf=conf_threshold,
                verbose=False,
                imgsz=640
            )
        except Exception as e:
            print(f"警告: 预测图片 {img_path.name} 时出错: {e}，跳过")
            continue
        
        # 读取图片尺寸
        try:
            img = Image.open(img_path)
            width, height = img.size
        except Exception as e:
            print(f"警告: 无法读取图片 {img_path.name}: {e}，跳过")
            continue
        
        # 处理预测结果
        if not results or len(results) == 0:
            continue
            
        for result in results:
            if not hasattr(result, 'boxes') or result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls)
                conf = float(box.conf)
                
                # YOLO格式: 归一化坐标 [x_center, y_center, width, height]
                xywhn = box.xywhn[0].cpu().numpy()
                x_center, y_center, w, h = xywhn
                
                # 转换为COCO格式: 绝对坐标 [x, y, width, height] (左上角)
                x = (x_center - w/2) * width
                y = (y_center - h/2) * height
                w_px = w * width
                h_px = h * height
                
                # 确保坐标在有效范围内
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w_px = max(1, min(w_px, width - x))
                h_px = max(1, min(h_px, height - y))
                
                # 获取image_id
                image_id = image_id_map[img_path.name]
                
                # 添加预测（包含score和category字段，这是SafeDNN-Clean需要的）
                predictions["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "category": class_names[class_id],  # SafeDNN-Clean需要
                    "bbox": [float(x), float(y), float(w_px), float(h_px)],
                    "area": float(w_px * h_px),
                    "score": float(conf),  # SafeDNN-Clean需要（置信度分数）
                    "iscrowd": 0
                })
                annotation_id += 1
                total_predictions += 1
        
        if i % 20 == 0:
            print(f"  已处理: {i}/{len(image_files)} 张图片, {total_predictions} 个预测")
    
    # 保存预测结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    processed_count = len(image_files) - skipped_count
    print(f"\n✅ 预测完成!")
    print(f"   总图片数: {len(image_files)}")
    print(f"   处理图片: {processed_count}")
    print(f"   跳过图片: {skipped_count} (不在标注文件中)")
    print(f"   预测数量: {total_predictions}")
    if processed_count > 0:
        print(f"   平均每张图片: {total_predictions/processed_count:.2f} 个预测")
    print(f"   保存到: {output_path.absolute()}")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="生成COCO格式的模型预测结果"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型权重路径"
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="val",
        help="验证集目录或split名称（train/val）"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="annotations_val_coco.json",
        help="COCO格式的标注文件（用于获取image_id映射）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions_coco.json",
        help="输出文件路径"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值 (默认: 0.25)"
    )
    
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

