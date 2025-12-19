#!/usr/bin/env python3
"""
可视化质量报告中的问题标注
1. 显示质量分数最低（最不可靠）的前N个标注
2. 按错误类型分类显示示例

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

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 颜色配置 (RGB格式)
COLORS = {
    "spurious": (255, 0, 0),       # 红色
    "location": (255, 165, 0),    # 橙色
    "label": (255, 0, 255),       # 紫色
    "missing": (255, 255, 0),     # 黄色
    "normal": (0, 255, 0),        # 绿色
}

# 类别名称
CATEGORY_NAMES = {
    0: "Carpetweed",
    1: "Morning Glory",
    2: "Palmer Amaranth"
}


def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def bbox_to_xyxy(bbox):
    """将COCO格式的bbox [x, y, w, h] 转换为 [x1, y1, x2, y2]"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def draw_bbox_with_label(draw, bbox, color, label, width=5):
    """绘制边界框和标签"""
    x1, y1, x2, y2 = bbox_to_xyxy(bbox)
    
    # 绘制边界框
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    
    # 绘制标签
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
    
    # 标签背景
    draw.rectangle(
        [x1, label_y, x1 + text_width + 20, label_y + text_height + 10],
        fill=(0, 0, 0, 240)
    )
    
    # 标签文字（白色）
    draw.text((x1 + 10, label_y + 5), label, fill=(255, 255, 255), font=font)


def visualize_lowest_quality(
    quality_report,
    annotations_data,
    images_dir,
    output_dir,
    top_n=20
):
    """可视化质量分数最低的前N个标注"""
    print("\n" + "=" * 70)
    print(f"可视化质量分数最低的前 {top_n} 个标注")
    print("=" * 70)
    
    # 获取所有GT标注（正ID）及其质量分数
    gt_annotations = []
    for ann in quality_report["annotations"]:
        if ann.get("id", 0) >= 0:  # GT标注
            quality = ann.get("quality", 1.0)
            issue = ann.get("issue", None)
            gt_annotations.append({
                "ann": ann,
                "quality": quality,
                "issue": issue
            })
    
    # 按质量分数排序（分数越低越不可靠）
    gt_annotations.sort(key=lambda x: x["quality"])
    
    # 取前N个
    top_annotations = gt_annotations[:top_n]
    
    print(f"找到 {len(top_annotations)} 个最不可靠的标注")
    
    # 创建输出目录
    output_path = Path(output_dir) / "lowest_quality"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建图像ID到文件名的映射
    image_id_to_info = {img['id']: img for img in annotations_data['images']}
    
    # 按图像组织标注
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
        
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # 绘制每个标注
        for item in items:
            ann = item["ann"]
            quality = item["quality"]
            issue = item["issue"]
            category_id = ann['category_id']
            category_name = CATEGORY_NAMES.get(category_id, f"Class {category_id}")
            
            # 根据错误类型选择颜色
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
        
        # 添加标题
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
        
        # 保存
        output_filename = f"{image_id:05d}_{image_filename}"
        output_file = output_path / output_filename
        img.save(output_file, quality=95)
        success_count += 1
    
    print(f"✅ 已生成 {success_count} 张图像到: {output_path}")
    return success_count


def visualize_by_issue_type(
    quality_report,
    annotations_data,
    images_dir,
    output_dir,
    samples_per_type=10
):
    """按错误类型可视化示例"""
    print("\n" + "=" * 70)
    print(f"按错误类型可视化示例（每种类型 {samples_per_type} 张）")
    print("=" * 70)
    
    # 按错误类型组织标注
    issues_by_type = {
        "spurious": [],
        "location": [],
        "label": [],
        "missing": []
    }
    
    # 收集GT标注
    for ann in quality_report["annotations"]:
        if ann.get("id", 0) >= 0:  # GT标注
            issue = ann.get("issue")
            if issue in issues_by_type:
                quality = ann.get("quality", 1.0)
                issues_by_type[issue].append({
                    "ann": ann,
                    "quality": quality
                })
    
    # 收集missing预测
    for ann in quality_report["annotations"]:
        if ann.get("id", 0) < 0:  # Missing预测（负ID）
            issues_by_type["missing"].append({
                "ann": ann,
                "quality": ann.get("quality", 0.0)
            })
    
    # 按质量分数排序（分数越低越严重）
    for issue_type in issues_by_type:
        issues_by_type[issue_type].sort(key=lambda x: x["quality"])
        print(f"  {issue_type}: {len(issues_by_type[issue_type])} 个问题")
    
    # 创建图像ID到文件名的映射
    image_id_to_info = {img['id']: img for img in annotations_data['images']}
    
    # 为每种错误类型生成可视化
    for issue_type, items in issues_by_type.items():
        if len(items) == 0:
            print(f"\n⚠️  {issue_type}: 没有找到问题")
            continue
        
        print(f"\n处理 {issue_type} 类型...")
        
        # 创建输出目录
        output_path = Path(output_dir) / issue_type
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 取前N个样本
        samples = items[:samples_per_type]
        
        # 按图像组织
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
            
            # 加载图像
            img = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(img)
            
            # 绘制每个标注
            for item in image_items:
                ann = item["ann"]
                quality = item["quality"]
                category_id = ann.get('category_id', -1)
                
                if issue_type == "missing":
                    # Missing是预测框，可能没有category_id
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
            
            # 添加标题
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
            
            # 保存
            output_filename = f"{image_id:05d}_{image_filename}"
            output_file = output_path / output_filename
            img.save(output_file, quality=95)
            success_count += 1
        
        print(f"  ✅ 已生成 {success_count} 张图像到: {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="可视化质量报告中的问题标注"
    )
    parser.add_argument(
        "--quality-report",
        type=str,
        default="quality_report_train.json",
        help="质量报告文件"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="annotations_train_coco.json",
        help="原始标注文件（COCO格式）"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="cotton weed dataset/train/images",
        help="图像目录路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="quality_issues_visualization",
        help="输出目录"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="显示质量分数最低的前N个标注（默认: 20）"
    )
    parser.add_argument(
        "--samples-per-type",
        type=int,
        default=10,
        help="每种错误类型显示的样本数（默认: 10）"
    )
    
    args = parser.parse_args()
    
    # 检查文件
    if not Path(args.quality_report).exists():
        print(f"ERROR: Quality report file not found: {args.quality_report}")
        return 1
    
    if not Path(args.annotations).exists():
        print(f"ERROR: Annotations file not found: {args.annotations}")
        return 1
    
    print("=" * 70)
    print("质量报告可视化工具")
    print("=" * 70)
    print(f"质量报告: {args.quality_report}")
    print(f"标注文件: {args.annotations}")
    print(f"图像目录: {args.images_dir}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 70)
    
    # 加载数据
    print("\n加载数据文件...")
    quality_report = load_json(args.quality_report)
    annotations_data = load_json(args.annotations)
    
    print(f"  质量报告标注数: {len(quality_report['annotations'])}")
    print(f"  图像数: {len(annotations_data['images'])}")
    
    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 可视化质量分数最低的标注
    visualize_lowest_quality(
        quality_report,
        annotations_data,
        args.images_dir,
        args.output_dir,
        args.top_n
    )
    
    # 2. 按错误类型可视化
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
    print(f"输出目录: {output_path.absolute()}")
    print("\n生成的文件夹:")
    print("  - lowest_quality/ (质量分数最低的前N个标注)")
    print("  - spurious/ (虚假标注示例)")
    print("  - location/ (定位错误示例)")
    print("  - label/ (类别错误示例)")
    print("  - missing/ (缺失标注示例)")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())

