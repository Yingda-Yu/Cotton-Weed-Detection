#!/usr/bin/env python3
"""
可视化数据清洗前后的变化
在图像上标注原始标注、清洗后标注和错误类型

Usage:
    python tools/visualize_cleaning_changes.py \
        --original annotations_train_coco.json \
        --cleaned cleaned_train_annotations.json \
        --quality-report quality_report_train.json \
        --output-dir cleaning_visualization
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 颜色配置 (RGB格式，用于PIL)
COLORS = {
    "original": (0, 0, 255),       # 红色 - 原始标注
    "cleaned": (0, 255, 0),        # 绿色 - 清洗后标注
    "spurious": (255, 0, 0),       # 蓝色 - 虚假标注（将被删除）
    "location": (255, 165, 0),     # 橙色 - 定位错误（已修复）
    "label": (255, 0, 255),        # 紫色 - 类别错误（已修复）
    "missing": (255, 255, 0),      # 黄色 - 缺失标注（已添加）
    "normal": (128, 128, 128),     # 灰色 - 正常标注（未修改）
    "background": (255, 255, 255)  # 白色 - 背景
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


def draw_bbox(draw, bbox, color, label="", width=4, label_position="top"):
    """在PIL ImageDraw上绘制边界框
    
    Args:
        draw: ImageDraw对象
        bbox: 边界框 [x, y, w, h]
        color: 颜色 (RGB)
        label: 标签文本
        width: 线宽
        label_position: 标签位置 "top" 或 "bottom"
    """
    x1, y1, x2, y2 = bbox_to_xyxy(bbox)
    
    # 绘制边界框（更粗的线）
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    
    # 绘制标签背景
    if label:
        # 尝试使用更大的字体
        font_size = 48  # 进一步增大字体
        try:
            # 尝试加载系统字体
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
                except:
                    # 如果都失败，使用默认字体但尝试放大
                    font = ImageFont.load_default()
        
        # 获取文本尺寸
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # 根据位置决定标签位置
        if label_position == "bottom":
            # 标签在框下方
            label_y = y2 + 5
        else:
            # 标签在框上方（默认）
            label_y = max(0, y1 - text_height - 8)
        
        # 绘制标签背景（更大的背景）
        draw.rectangle(
            [x1, label_y, x1 + text_width + 16, label_y + text_height + 8],
            fill=(0, 0, 0, 220)  # 更不透明的背景
        )
        
        # 绘制标签文本（使用白色，更清晰可见）
        draw.text(
            (x1 + 8, label_y + 4),
            label,
            fill=(255, 255, 255),  # 白色文字
            font=font
        )


def create_comparison_image(
    image_path,
    original_anns,
    cleaned_anns,
    quality_report_anns,
    image_info,
    output_path
):
    """创建对比图像，显示原始和清洗后的标注"""
    
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    image_id = image_info['id']
    
    # 按annotation ID组织标注
    original_by_id = {ann['id']: ann for ann in original_anns if ann['image_id'] == image_id}
    cleaned_by_id = {ann['id']: ann for ann in cleaned_anns if ann['image_id'] == image_id}
    quality_by_id = {ann.get('id', -1): ann for ann in quality_report_anns 
                     if ann.get('image_id') == image_id}
    
    # 收集所有相关的标注ID
    all_ids = set(original_by_id.keys()) | set(cleaned_by_id.keys())
    
    # 第一步：绘制原始标注（红色，粗线，带错误类型标签）
    for ann_id in all_ids:
        if ann_id in original_by_id:
            ann = original_by_id[ann_id]
            category_id = ann['category_id']
            category_name = CATEGORY_NAMES.get(category_id, f"Class {category_id}")
            
            # 检查是否有质量问题
            quality_ann = quality_by_id.get(ann_id)
            issue = quality_ann.get('issue') if quality_ann else None
            
            if issue:
                # 根据错误类型选择颜色和标签
                if issue == 'spurious':
                    color = COLORS['spurious']
                    label = f"ORIGINAL: {category_name} [SPURIOUS - WILL DELETE]"
                elif issue == 'location':
                    color = COLORS['location']
                    label = f"ORIGINAL: {category_name} [LOCATION ERROR]"
                elif issue == 'label':
                    color = COLORS['label']
                    label = f"ORIGINAL: {category_name} [LABEL ERROR]"
                else:
                    color = COLORS['original']
                    label = f"ORIGINAL: {category_name}"
            else:
                color = COLORS['original']
                label = f"ORIGINAL: {category_name}"
            
            # 绘制原始标注（红色粗线，标签在框下方）
            draw_bbox(draw, ann['bbox'], color, label, width=5, label_position="bottom")
    
    # 第二步：绘制清洗后的标注（绿色，更粗的线，显示在原始标注上方）
    # 注意：missing的情况不在这里处理，因为没有对应的原始标注
    for ann_id in all_ids:
        if ann_id in cleaned_by_id:
            ann = cleaned_by_id[ann_id]
            category_id = ann['category_id']
            category_name = CATEGORY_NAMES.get(category_id, f"Class {category_id}")
            
            # 检查原始标注是否存在
            original_ann = original_by_id.get(ann_id)
            
            # 如果不存在原始标注，说明是missing（新添加的），跳过这里，后面单独处理
            if not original_ann:
                continue
            
            if original_ann:
                # 检查是否有变化
                original_cat = original_ann['category_id']
                original_bbox = original_ann['bbox']
                cleaned_cat = ann['category_id']
                cleaned_bbox = ann['bbox']
                
                bbox_changed = original_bbox != cleaned_bbox
                cat_changed = original_cat != cleaned_cat
                
                if bbox_changed or cat_changed:
                    # 有变化，用绿色粗线绘制，标签显示在框的上方
                    if bbox_changed and cat_changed:
                        label = f"CLEANED: {category_name} [BOTH FIXED]"
                    elif bbox_changed:
                        label = f"CLEANED: {category_name} [LOCATION FIXED]"
                    elif cat_changed:
                        label = f"CLEANED: {category_name} [LABEL FIXED]"
                    else:
                        label = f"CLEANED: {category_name}"
                    # 使用更粗的实线，让清洗后的标注更明显，标签在框上方
                    x1, y1, x2, y2 = bbox_to_xyxy(ann['bbox'])
                    # 绘制实线框（与原始标注区分）
                    draw.rectangle([x1, y1, x2, y2], outline=COLORS['cleaned'], width=8)
                    
                    # 标签显示在框的上方
                    try:
                        font = ImageFont.truetype("arial.ttf", 48)
                    except:
                        try:
                            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 48)
                        except:
                            font = ImageFont.load_default()
                    
                    bbox_text = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox_text[2] - bbox_text[0]
                    text_height = bbox_text[3] - bbox_text[1]
                    label_y = max(0, y1 - text_height - 12)  # 在框上方
                    draw.rectangle(
                        [x1, label_y, x1 + text_width + 20, label_y + text_height + 12],
                        fill=(0, 0, 0, 240)
                    )
                    # 使用白色文字
                    draw.text((x1 + 10, label_y + 6), label, fill=(255, 255, 255), font=font)
                else:
                    # 没有变化，但仍然显示清洗后的标注
                    label = f"CLEANED: {category_name} [NO CHANGE]"
                    x1, y1, x2, y2 = bbox_to_xyxy(ann['bbox'])
                    draw.rectangle([x1, y1, x2, y2], outline=COLORS['cleaned'], width=6)
                    
                    try:
                        font = ImageFont.truetype("arial.ttf", 48)
                    except:
                        try:
                            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 48)
                        except:
                            font = ImageFont.load_default()
                    
                    bbox_text = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox_text[2] - bbox_text[0]
                    text_height = bbox_text[3] - bbox_text[1]
                    label_y = max(0, y1 - text_height - 12)
                    draw.rectangle(
                        [x1, label_y, x1 + text_width + 20, label_y + text_height + 12],
                        fill=(0, 0, 0, 240)
                    )
                    # 使用白色文字
                    draw.text((x1 + 10, label_y + 6), label, fill=(255, 255, 255), font=font)
    
    # 第三步：单独处理missing标注（新添加的，没有原始标注）
    for ann_id in cleaned_by_id:
        if ann_id not in original_by_id:
            # 这是missing标注，只显示ADDED，不显示ORIGINAL
            ann = cleaned_by_id[ann_id]
            category_id = ann['category_id']
            category_name = CATEGORY_NAMES.get(category_id, f"Class {category_id}")
            label = f"ADDED: {category_name} [MISSING - NEW]"
            x1, y1, x2, y2 = bbox_to_xyxy(ann['bbox'])
            draw.rectangle([x1, y1, x2, y2], outline=COLORS['missing'], width=8)
            
            try:
                font = ImageFont.truetype("arial.ttf", 48)
            except:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 48)
                except:
                    font = ImageFont.load_default()
            
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            label_y = max(0, y1 - text_height - 12)
            draw.rectangle(
                [x1, label_y, x1 + text_width + 20, label_y + text_height + 12],
                fill=(0, 0, 0, 240)
            )
            # 使用白色文字
            draw.text((x1 + 10, label_y + 6), label, fill=(255, 255, 255), font=font)
    
    # 第三步：绘制被删除的标注（spurious，用虚线效果）
    for ann_id in all_ids:
        if ann_id in original_by_id and ann_id not in cleaned_by_id:
            ann = original_by_id[ann_id]
            category_id = ann['category_id']
            category_name = CATEGORY_NAMES.get(category_id, f"Class {category_id}")
            
            # 检查是否是spurious
            quality_ann = quality_by_id.get(ann_id)
            issue = quality_ann.get('issue') if quality_ann else None
            
            if issue == 'spurious':
                # 绘制被删除的标注（用虚线效果，通过多次绘制实现）
                x1, y1, x2, y2 = bbox_to_xyxy(ann['bbox'])
                # 绘制虚线框
                dash_length = 10
                gap_length = 5
                # 上边
                for x in range(int(x1), int(x2), dash_length + gap_length):
                    draw.line([(x, y1), (min(x + dash_length, x2), y1)], fill=COLORS['spurious'], width=5)
                # 下边
                for x in range(int(x1), int(x2), dash_length + gap_length):
                    draw.line([(x, y2), (min(x + dash_length, x2), y2)], fill=COLORS['spurious'], width=5)
                # 左边
                for y in range(int(y1), int(y2), dash_length + gap_length):
                    draw.line([(x1, y), (x1, min(y + dash_length, y2))], fill=COLORS['spurious'], width=5)
                # 右边
                for y in range(int(y1), int(y2), dash_length + gap_length):
                    draw.line([(x2, y), (x2, min(y + dash_length, y2))], fill=COLORS['spurious'], width=5)
                
                # 添加删除标签
                label = f"DELETED: {category_name} [SPURIOUS]"
                try:
                    font = ImageFont.truetype("arial.ttf", 32)
                except:
                    try:
                        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 32)
                    except:
                        font = ImageFont.load_default()
                
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
                label_y = max(0, y1 - text_height - 8)
                draw.rectangle(
                    [x1, label_y, x1 + text_width + 16, label_y + text_height + 8],
                    fill=(0, 0, 0, 220)
                )
                draw.text((x1 + 8, label_y + 4), label, fill=COLORS['spurious'], font=font)
    
    # 添加图例（更大的字体）
    legend_y = 10
    legend_items = [
        ("Original (Red)", COLORS['original']),
        ("Cleaned (Green)", COLORS['cleaned']),
        ("Spurious (Blue)", COLORS['spurious']),
        ("Location Error (Orange)", COLORS['location']),
        ("Label Error (Purple)", COLORS['label']),
        ("Missing (Yellow)", COLORS['missing']),
    ]
    
    # 使用更大的字体
    legend_font_size = 40
    try:
        legend_font = ImageFont.truetype("arial.ttf", legend_font_size)
    except:
        try:
            legend_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", legend_font_size)
        except:
            try:
                legend_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", legend_font_size)
            except:
                legend_font = ImageFont.load_default()
    
    for i, (text, color) in enumerate(legend_items):
        y_pos = legend_y + i * 55  # 增加间距
        # 绘制颜色块（更大）
        draw.rectangle([10, y_pos, 50, y_pos + 35], fill=color)
        # 绘制文本（更大，加粗边框）
        draw.text((60, y_pos), text, fill=(255, 255, 255), font=legend_font, 
                 stroke_width=4, stroke_fill=(0, 0, 0))
    
    # 保存图像
    img.save(output_path, quality=95)
    return output_path


def visualize_cleaning_changes(
    original_file,
    cleaned_file,
    quality_report_file,
    images_dir,
    output_dir,
    dataset_yaml="dataset.yaml"
):
    """
    可视化数据清洗前后的变化
    
    Args:
        original_file: 原始标注文件（COCO格式）
        cleaned_file: 清洗后的标注文件（COCO格式）
        quality_report_file: 质量报告文件
        images_dir: 图像目录路径
        output_dir: 输出目录
        dataset_yaml: 数据集配置文件（用于获取图像路径）
    """
    print("=" * 70)
    print("数据清洗可视化工具")
    print("=" * 70)
    
    # 加载数据
    print("\n[1/5] 加载数据文件...")
    original_data = load_json(original_file)
    cleaned_data = load_json(cleaned_file)
    quality_report = load_json(quality_report_file)
    
    print(f"   原始标注数: {len(original_data['annotations'])}")
    print(f"   清洗后标注数: {len(cleaned_data['annotations'])}")
    print(f"   质量报告标注数: {len(quality_report['annotations'])}")
    
    # 获取图像目录
    images_path = Path(images_dir)
    if not images_path.exists():
        # 尝试从dataset.yaml获取路径
        import yaml
        with open(dataset_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)
        images_path = Path(dataset_config.get('path', '.')) / dataset_config.get('train', 'train')
        if not images_path.exists():
            print(f"ERROR: Image directory not found: {images_path}")
            return False
    
    print(f"   图像目录: {images_path}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"   输出目录: {output_path}")
    
    # 创建按错误类型分类的子目录
    issue_dirs = {
        "spurious": output_path / "spurious",
        "location": output_path / "location",
        "label": output_path / "label",
        "missing": output_path / "missing"
    }
    for issue_dir in issue_dirs.values():
        issue_dir.mkdir(parents=True, exist_ok=True)
    print(f"   创建分类目录: {', '.join([d.name for d in issue_dirs.values()])}")
    
    # 按图像组织标注
    print("\n[2/5] 组织标注数据...")
    original_by_image = defaultdict(list)
    cleaned_by_image = defaultdict(list)
    quality_by_image = defaultdict(list)
    
    for ann in original_data['annotations']:
        original_by_image[ann['image_id']].append(ann)
    
    for ann in cleaned_data['annotations']:
        cleaned_by_image[ann['image_id']].append(ann)
    
    for ann in quality_report['annotations']:
        image_id = ann.get('image_id')
        if image_id is not None and image_id >= 0:  # 只处理正ID（GT标注），负ID是missing预测
            quality_by_image[image_id].append(ann)
    
    # 处理missing预测（负ID）
    missing_predictions = [ann for ann in quality_report['annotations'] 
                          if ann.get('id', 0) < 0]
    print(f"   Missing预测数: {len(missing_predictions)}")
    
    # 找出有变化的图像
    print("\n[3/5] 识别有变化的图像...")
    changed_images = set()
    
    # 检查每个原始标注
    for ann in original_data['annotations']:
        image_id = ann['image_id']
        ann_id = ann['id']
        
        # 检查是否有质量问题
        quality_anns = quality_by_image.get(image_id, [])
        has_issue = False
        for qa in quality_anns:
            if qa.get('id') == ann_id and qa.get('issue'):
                has_issue = True
                break
        
        if has_issue:
            changed_images.add(image_id)
    
    # 检查清洗后的标注（可能有新增的）
    for ann in cleaned_data['annotations']:
        image_id = ann['image_id']
        ann_id = ann['id']
        
        # 检查是否在原始标注中
        original_anns = original_by_image.get(image_id, [])
        found = False
        for oa in original_anns:
            if oa['id'] == ann_id:
                # 检查是否有变化
                if oa['category_id'] != ann['category_id'] or oa['bbox'] != ann['bbox']:
                    changed_images.add(image_id)
                    found = True
                    break
        if not found:
            # 新添加的标注
            changed_images.add(image_id)
    
    print(f"   有变化的图像数: {len(changed_images)}")
    
    # 创建图像ID到文件名的映射
    print("\n[4/5] 创建图像映射...")
    image_id_to_info = {img['id']: img for img in original_data['images']}
    
    # 生成可视化
    print("\n[5/5] 生成可视化图像...")
    success_count = 0
    error_count = 0
    issue_counts = {"spurious": 0, "location": 0, "label": 0, "missing": 0}
    
    for image_id in sorted(changed_images):
        image_info = image_id_to_info.get(image_id)
        if not image_info:
            continue
        
        image_filename = image_info['file_name']
        image_path = images_path / image_filename
        
        # 如果文件不存在，尝试查找相似的文件名（处理下划线数量差异）
        if not image_path.exists():
            # 尝试查找文件名（忽略下划线数量）
            base_name = image_filename.replace('_', '').replace('.jpg', '').replace('.JPG', '')
            found = False
            for actual_file in images_path.glob('*.jpg'):
                actual_base = actual_file.stem.replace('_', '')
                if actual_base == base_name:
                    image_path = actual_file
                    found = True
                    break
            
            if not found:
                # 尝试查找所有可能的变体
                for actual_file in images_path.glob('*.jpg'):
                    if image_filename.lower() in actual_file.name.lower() or actual_file.name.lower() in image_filename.lower():
                        image_path = actual_file
                        found = True
                        break
            
            if not found:
                # 最后尝试：直接查找文件名（不区分大小写）
                for actual_file in images_path.glob('*.jpg'):
                    if actual_file.name.lower() == image_filename.lower():
                        image_path = actual_file
                        found = True
                        break
            
            if not found:
                if error_count < 10:  # 只显示前10个错误
                    print(f"WARNING: Image not found: {image_filename}")
                error_count += 1
                continue
        
        # 获取该图像的所有标注
        original_anns = original_by_image.get(image_id, [])
        cleaned_anns = cleaned_by_image.get(image_id, [])
        quality_anns = quality_by_image.get(image_id, [])
        
        # 添加该图像的missing预测
        image_missing = [ann for ann in missing_predictions 
                        if ann.get('image_id') == image_id]
        quality_anns.extend(image_missing)
        
        # 分析该图像包含的错误类型（只统计实际被修复的）
        image_issues = set()
        
        # 检查每个原始标注是否真的被修复了
        for ann in original_anns:
            ann_id = ann['id']
            quality_ann = next((qa for qa in quality_anns if qa.get('id') == ann_id), None)
            
            if not quality_ann or not quality_ann.get('issue'):
                continue
            
            issue = quality_ann.get('issue')
            
            # 对于spurious，如果还在cleaned中，说明没被删除（不应该发生）
            if issue == 'spurious':
                if ann_id not in [ca['id'] for ca in cleaned_anns]:
                    image_issues.add('spurious')
                continue
            
            # 对于location和label，检查清洗后是否真的有变化
            cleaned_ann = next((ca for ca in cleaned_anns if ca['id'] == ann_id), None)
            if not cleaned_ann:
                continue  # 如果清洗后不存在，跳过
            
            if issue == 'location':
                # 检查bbox是否真的改变了
                original_bbox = ann['bbox']
                cleaned_bbox = cleaned_ann['bbox']
                if original_bbox != cleaned_bbox:
                    image_issues.add('location')
            
            elif issue == 'label':
                # 检查类别是否真的改变了
                original_cat = ann['category_id']
                cleaned_cat = cleaned_ann['category_id']
                if original_cat != cleaned_cat:
                    image_issues.add('label')
        
        # 检查是否有missing（新添加的标注）
        for ann in cleaned_anns:
            ann_id = ann['id']
            # 如果这个标注不在原始标注中，说明是missing
            if ann_id not in [oa['id'] for oa in original_anns]:
                image_issues.add('missing')
                break
        
        # 生成输出文件名
        output_filename = f"{image_id:05d}_{image_filename}"
        
        # 根据错误类型复制到对应文件夹
        if image_issues:
            # 如果包含多种错误类型，复制到所有相关文件夹
            for issue in image_issues:
                if issue in issue_dirs:
                    output_file = issue_dirs[issue] / output_filename
                    try:
                        create_comparison_image(
                            image_path,
                            original_anns,
                            cleaned_anns,
                            quality_anns,
                            image_info,
                            output_file
                        )
                        issue_counts[issue] += 1
                    except Exception as e:
                        print(f"ERROR: Failed to process image {image_filename} for {issue}: {e}")
                        error_count += 1
        else:
            # 没有明确错误类型，保存到主目录
            output_file = output_path / output_filename
            try:
                create_comparison_image(
                    image_path,
                    original_anns,
                    cleaned_anns,
                    quality_anns,
                    image_info,
                    output_file
                )
            except Exception as e:
                print(f"ERROR: Failed to process image {image_filename}: {e}")
                error_count += 1
        
        success_count += 1
        if success_count % 10 == 0:
            print(f"   已处理: {success_count} 张图像...")
    
    # 生成摘要报告
    print("\n" + "=" * 70)
    print("Visualization Complete!")
    print("=" * 70)
    print(f"SUCCESS: Processed {success_count} images")
    if error_count > 0:
        print(f"WARNING: Failed to process {error_count} images")
    print(f"\nOutput directory: {output_path.absolute()}")
    print("\n分类统计:")
    print(f"  - Spurious (虚假标注): {issue_counts['spurious']} 张图像")
    print(f"  - Location (定位错误): {issue_counts['location']} 张图像")
    print(f"  - Label (类别错误): {issue_counts['label']} 张图像")
    print(f"  - Missing (缺失标注): {issue_counts['missing']} 张图像")
    print("\n分类目录:")
    for issue, issue_dir in issue_dirs.items():
        count = len(list(issue_dir.glob("*.jpg")))
        print(f"  - {issue_dir.name}/: {count} 张图像")
    print("\n图例说明:")
    print("  - 红色虚线框: 原始标注（有问题）")
    print("  - 绿色实线框: 清洗后的标注（已修复）")
    print("  - Spurious: 虚假标注，已删除")
    print("  - Location: 定位错误，已修复位置")
    print("  - Label: 类别错误，已修复类别")
    print("  - Missing: 缺失标注，已添加")
    print("=" * 70)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="可视化数据清洗前后的变化"
    )
    parser.add_argument(
        "--original",
        type=str,
        default="annotations_train_coco.json",
        help="原始标注文件（COCO格式）"
    )
    parser.add_argument(
        "--cleaned",
        type=str,
        default="cleaned_train_annotations.json",
        help="清洗后的标注文件（COCO格式）"
    )
    parser.add_argument(
        "--quality-report",
        type=str,
        default="quality_report_train.json",
        help="质量报告文件"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="cotton weed dataset/train",
        help="图像目录路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cleaning_visualization",
        help="输出目录"
    )
    parser.add_argument(
        "--dataset-yaml",
        type=str,
        default="dataset.yaml",
        help="数据集配置文件（用于获取图像路径）"
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.original).exists():
        print(f"ERROR: Original annotation file not found: {args.original}")
        return 1
    
    if not Path(args.cleaned).exists():
        print(f"ERROR: Cleaned annotation file not found: {args.cleaned}")
        return 1
    
    if not Path(args.quality_report).exists():
        print(f"ERROR: Quality report file not found: {args.quality_report}")
        return 1
    
    success = visualize_cleaning_changes(
        args.original,
        args.cleaned,
        args.quality_report,
        args.images_dir,
        args.output_dir,
        args.dataset_yaml
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

