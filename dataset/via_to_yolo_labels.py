#!/usr/bin/env python3
"""
将VIA格式的annotations转换为YOLO格式的labels

用法:
    python dataset/via_to_yolo_labels.py
"""

import json
import yaml
from pathlib import Path
from PIL import Image
from collections import defaultdict

# ================================
# 配置
# ================================
WORKSPACE_ROOT = Path(__file__).parent.parent
DATASET_ROOT = WORKSPACE_ROOT / "cotton weed dataset"

# 类别映射
CLASS_NAME_TO_ID = {
    "carpetweed": 0,
    "morningglory": 1,
    "palmer_amaranth": 2
}


def via_to_yolo_label(via_file: Path, images_dir: Path, output_labels_dir: Path) -> bool:
    """
    将单个VIA格式文件转换为YOLO格式标签
    
    Args:
        via_file: VIA格式JSON文件路径
        images_dir: 图片目录
        output_labels_dir: 输出labels目录
    
    Returns:
        是否成功转换
    """
    try:
        # 读取VIA格式文件
        with open(via_file, 'r', encoding='utf-8') as f:
            via_data = json.load(f)
        
        # 获取第一个key（VIA格式通常只有一个key）
        via_key = list(via_data.keys())[0]
        file_info = via_data[via_key]
        
        # 获取文件名
        filename = file_info.get("filename", "")
        if not filename:
            return False
        
        # 查找对应的图片文件
        base_name = Path(filename).stem
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            potential = images_dir / (base_name + ext)
            if potential.exists():
                img_path = potential
                break
        
        if img_path is None:
            print(f"警告: 找不到图片文件: {filename}")
            return False
        
        # 获取图片尺寸
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"警告: 无法读取图片 {img_path}: {e}")
            return False
        
        # 转换标注
        regions = file_info.get("regions", [])
        yolo_lines = []
        
        for region in regions:
            shape_attrs = region.get("shape_attributes", {})
            region_attrs = region.get("region_attributes", {})
            
            # 只处理矩形标注
            if shape_attrs.get("name") != "rect":
                continue
            
            # 获取边界框坐标
            x = shape_attrs.get("x", 0)
            y = shape_attrs.get("y", 0)
            w = shape_attrs.get("width", 0)
            h = shape_attrs.get("height", 0)
            
            # 获取类别
            class_name = region_attrs.get("class", "")
            class_id = CLASS_NAME_TO_ID.get(class_name.lower(), None)
            
            if class_id is None:
                print(f"警告: 未知类别 '{class_name}'，跳过")
                continue
            
            # 转换为YOLO格式（归一化）
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            
            # 确保在[0, 1]范围内
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))
            
            # YOLO格式: class_id x_center y_center width height
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # 保存YOLO格式文件
        label_file = output_labels_dir / f"{base_name}.txt"
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        with open(label_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))
        
        return True
        
    except Exception as e:
        print(f"错误: 处理 {via_file} 时出错: {e}")
        return False


def convert_split(split: str):
    """
    转换指定split的annotations
    
    Args:
        split: "train" 或 "val"
    """
    print("=" * 70)
    print(f"转换 {split} 集的annotations为YOLO格式labels")
    print("=" * 70)
    
    annotations_dir = DATASET_ROOT / split / "annotations"
    images_dir = DATASET_ROOT / split / "images"
    labels_dir = DATASET_ROOT / split / "labels"
    
    # 检查目录
    if not annotations_dir.exists():
        print(f"❌ 错误: annotations目录不存在: {annotations_dir}")
        return
    
    if not images_dir.exists():
        print(f"❌ 错误: images目录不存在: {images_dir}")
        return
    
    # 创建labels目录
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有JSON文件
    via_files = list(annotations_dir.glob("*.json"))
    print(f"\n找到 {len(via_files)} 个annotation文件")
    
    # 转换统计
    converted = 0
    skipped = 0
    class_counts = defaultdict(int)
    
    # 转换每个文件
    for i, via_file in enumerate(via_files, 1):
        if via_to_yolo_label(via_file, images_dir, labels_dir):
            converted += 1
            # 统计类别
            try:
                with open(via_file, 'r', encoding='utf-8') as f:
                    via_data = json.load(f)
                    via_key = list(via_data.keys())[0]
                    regions = via_data[via_key].get("regions", [])
                    for region in regions:
                        class_name = region.get("region_attributes", {}).get("class", "")
                        if class_name:
                            class_counts[class_name.lower()] += 1
            except:
                pass
        else:
            skipped += 1
        
        if i % 50 == 0:
            print(f"  已处理: {i}/{len(via_files)}")
    
    # 打印统计信息
    print("\n" + "=" * 70)
    print("转换完成！")
    print("=" * 70)
    print(f"  成功转换: {converted} 个文件")
    print(f"  跳过: {skipped} 个文件")
    print(f"  输出目录: {labels_dir.absolute()}")
    
    if class_counts:
        print(f"\n  类别统计:")
        for class_name, count in sorted(class_counts.items()):
            print(f"    {class_name}: {count} 个标注")
    
    print("=" * 70)


def main():
    """主函数"""
    print("=" * 70)
    print("VIA格式annotations转YOLO格式labels")
    print("=" * 70)
    
    # 转换train和val
    for split in ["train", "val"]:
        convert_split(split)
        print()
    
    print("✅ 所有转换完成！")
    print(f"\n数据集结构:")
    print(f"  {DATASET_ROOT / 'train' / 'labels'}")
    print(f"  {DATASET_ROOT / 'val' / 'labels'}")


if __name__ == "__main__":
    main()

