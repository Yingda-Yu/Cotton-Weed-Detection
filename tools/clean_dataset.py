#!/usr/bin/env python3
"""
自动清洗数据集标注
基于SafeDNN-Clean的质量报告自动修复4种错误类型

⚠️ 注意：此脚本不会修改原始数据集，所有输出保存到新文件
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

def bbox_iou(bbox1, bbox2):
    """计算两个边界框的IoU"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # 计算交集
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

def find_matching_prediction(gt_ann, predictions_by_image, iou_threshold=0.3):
    """为GT标注找到匹配的预测框"""
    image_id = gt_ann["image_id"]
    if image_id not in predictions_by_image:
        return None
    
    gt_bbox = gt_ann["bbox"]
    best_match = None
    best_iou = 0.0
    
    for pred in predictions_by_image[image_id]:
        iou = bbox_iou(gt_bbox, pred["bbox"])
        if iou > best_iou and iou >= iou_threshold:
            best_iou = iou
            best_match = pred
    
    return best_match

def clean_dataset(
    quality_report_file="quality_report.json",
    predictions_file="predictions_coco.json",
    output_file="cleaned_annotations.json",
    spurious_threshold=0.3,
    location_score_threshold=0.7,
    label_score_threshold=0.8,
    missing_score_threshold=0.5
):
    """
    清洗数据集标注
    
    Args:
        quality_report_file: SafeDNN-Clean生成的质量报告
        predictions_file: 模型预测结果（用于location/label匹配）
        output_file: 输出文件（不会覆盖原始数据）
        spurious_threshold: spurious删除的阈值（暂不使用，直接删除所有spurious）
        location_score_threshold: location修复的预测分数阈值
        label_score_threshold: label修复的预测分数阈值
        missing_score_threshold: missing添加的预测分数阈值
    """
    print("=" * 70)
    print("数据集自动清洗")
    print("=" * 70)
    print(f"⚠️  原始数据集不会被修改，清洗结果保存到: {output_file}")
    print("=" * 70)
    
    # 1. 加载质量报告
    print(f"\n[1/5] 加载质量报告: {quality_report_file}")
    with open(quality_report_file, 'r', encoding='utf-8') as f:
        quality_report = json.load(f)
    
    # 2. 加载预测结果（用于location/label匹配）
    print(f"[2/5] 加载预测结果: {predictions_file}")
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)
    
    # 按image_id组织预测框
    predictions_by_image = defaultdict(list)
    for pred in predictions_data["annotations"]:
        predictions_by_image[pred["image_id"]].append(pred)
    
    print(f"   预测框总数: {len(predictions_data['annotations'])}")
    
    # 3. 分离不同类型的标注
    print(f"\n[3/5] 分析标注...")
    all_annotations = quality_report["annotations"]
    
    # 分离GT标注（正ID）和missing预测（负ID）
    gt_annotations = [ann for ann in all_annotations if ann.get("id", 0) >= 0]
    missing_predictions = [ann for ann in all_annotations if ann.get("id", 0) < 0]
    
    print(f"   GT标注总数: {len(gt_annotations)}")
    print(f"   Missing预测: {len(missing_predictions)}")
    
    # 统计问题类型
    issue_stats = defaultdict(int)
    for ann in gt_annotations:
        issue = ann.get("issue")
        if issue:
            issue_stats[issue] += 1
    
    print(f"\n   问题统计:")
    print(f"     Spurious (虚假标注): {issue_stats['spurious']}")
    print(f"     Location (定位错误): {issue_stats['location']}")
    print(f"     Label (类别错误): {issue_stats['label']}")
    print(f"     Missing (缺失标注): {len(missing_predictions)}")
    
    # 4. 清洗标注
    print(f"\n[4/5] 执行清洗...")
    cleaned_annotations = []
    stats = {
        "deleted_spurious": 0,
        "added_missing": 0,
        "fixed_location": 0,
        "fixed_label": 0,
        "kept_normal": 0,
        "skipped_location": 0,
        "skipped_label": 0,
        "skipped_missing": 0
    }
    
    new_id_counter = 0
    
    # 处理GT标注
    for ann in gt_annotations:
        issue = ann.get("issue")
        
        # 1. 删除 spurious
        if issue == "spurious":
            stats["deleted_spurious"] += 1
            continue  # 不添加到输出
        
        # 2. 修复 location（替换bbox）
        if issue == "location":
            # 找到匹配的预测框
            matching_pred = find_matching_prediction(ann, predictions_by_image)
            
            if matching_pred and matching_pred.get("score", 0) >= location_score_threshold:
                # 用预测框的bbox替换GT的bbox
                ann["bbox"] = matching_pred["bbox"]
                ann["area"] = matching_pred["bbox"][2] * matching_pred["bbox"][3]
                stats["fixed_location"] += 1
            else:
                # 没有找到匹配或分数不够，保留原标注
                stats["skipped_location"] += 1
                stats["kept_normal"] += 1
            
            # 清理临时字段
            ann.pop("issue", None)
            ann.pop("quality", None)
            ann.pop("cluster", None)
            ann.pop("category", None)  # 保留category_id
            cleaned_annotations.append(ann)
            continue
        
        # 3. 修复 label（替换类别）
        if issue == "label":
            # 找到匹配的预测框
            matching_pred = find_matching_prediction(ann, predictions_by_image)
            
            if matching_pred and matching_pred.get("score", 0) >= label_score_threshold:
                # 用预测框的类别替换GT的类别
                ann["category_id"] = matching_pred["category_id"]
                stats["fixed_label"] += 1
            else:
                # 没有找到匹配或分数不够，保留原标注
                stats["skipped_label"] += 1
                stats["kept_normal"] += 1
            
            # 清理临时字段
            ann.pop("issue", None)
            ann.pop("quality", None)
            ann.pop("cluster", None)
            ann.pop("category", None)
            cleaned_annotations.append(ann)
            continue
        
        # 4. 正常标注（无issue或issue为其他值）
        # 清理临时字段
        ann.pop("issue", None)
        ann.pop("quality", None)
        ann.pop("cluster", None)
        ann.pop("category", None)
        cleaned_annotations.append(ann)
        stats["kept_normal"] += 1
    
    # 5. 添加 missing 预测框
    print(f"\n[5/5] 添加missing标注...")
    for pred in missing_predictions:
        score = pred.get("score", 0)
        
        if score >= missing_score_threshold:
            # 创建新的GT标注
            new_ann = {
                "id": new_id_counter,
                "image_id": pred["image_id"],
                "category_id": pred["category_id"],
                "bbox": pred["bbox"],
                "area": pred["area"],
                "iscrowd": 0
            }
            cleaned_annotations.append(new_ann)
            stats["added_missing"] += 1
            new_id_counter += 1
        else:
            stats["skipped_missing"] += 1
    
    # 重新分配ID（确保连续）
    for i, ann in enumerate(cleaned_annotations):
        ann["id"] = i
    
    # 6. 构建输出
    output_data = {
        "info": quality_report["info"],
        "licenses": quality_report["licenses"],
        "images": quality_report["images"],
        "annotations": cleaned_annotations,
        "categories": quality_report["categories"]
    }
    
    # 7. 保存结果
    print(f"\n[保存] 保存清洗后的标注到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 8. 打印统计
    print("\n" + "=" * 70)
    print("清洗完成！统计信息：")
    print("=" * 70)
    print(f"  原始GT标注数: {len(gt_annotations)}")
    print(f"  清洗后标注数: {len(cleaned_annotations)}")
    print(f"\n  操作统计:")
    print(f"    ✓ 删除 spurious: {stats['deleted_spurious']}")
    print(f"    ✓ 修复 location: {stats['fixed_location']}")
    print(f"    ✓ 修复 label: {stats['fixed_label']}")
    print(f"    ✓ 添加 missing: {stats['added_missing']}")
    print(f"    ✓ 保留正常: {stats['kept_normal']}")
    print(f"\n  跳过统计:")
    print(f"    ⚠ 跳过 location (无匹配/分数低): {stats['skipped_location']}")
    print(f"    ⚠ 跳过 label (无匹配/分数低): {stats['skipped_label']}")
    print(f"    ⚠ 跳过 missing (分数低): {stats['skipped_missing']}")
    print(f"\n  净变化: {len(cleaned_annotations) - len(gt_annotations)} 个标注")
    print(f"\n  ✅ 原始数据集未修改，清洗结果已保存到: {output_file}")
    print("=" * 70)
    
    return output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动清洗数据集标注（保留原始数据）")
    parser.add_argument("--quality-report", type=str, default="quality_report.json",
                        help="质量报告文件路径")
    parser.add_argument("--predictions", type=str, default="predictions_coco.json",
                        help="预测结果文件路径")
    parser.add_argument("--output", type=str, default="cleaned_annotations.json",
                        help="输出文件路径（不会覆盖原始数据）")
    parser.add_argument("--location-threshold", type=float, default=0.7,
                        help="Location修复的预测分数阈值 (默认: 0.7)")
    parser.add_argument("--label-threshold", type=float, default=0.8,
                        help="Label修复的预测分数阈值 (默认: 0.8)")
    parser.add_argument("--missing-threshold", type=float, default=0.5,
                        help="Missing添加的预测分数阈值 (默认: 0.5)")
    
    args = parser.parse_args()
    
    clean_dataset(
        quality_report_file=args.quality_report,
        predictions_file=args.predictions,
        output_file=args.output,
        location_score_threshold=args.location_threshold,
        label_score_threshold=args.label_threshold,
        missing_score_threshold=args.missing_threshold
    )

