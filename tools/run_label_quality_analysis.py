#!/usr/bin/env python3
"""
完整的标签质量分析流程
整合YOLO预测、COCO转换、SafeDNN-Clean分析

用法:
    python run_label_quality_analysis.py \
        --model runs/detect/yolov8n_baseline/weights/best.pt \
        --split val
"""

import subprocess
import json
import argparse
from pathlib import Path
import sys


def run_analysis_pipeline(
    model_weights,
    split="val",
    iou_threshold=0.5,
    quality_threshold=0.5,
    conf_threshold=0.25
):
    """
    运行完整的标签质量分析流程
    
    Args:
        model_weights: 模型权重路径
        split: 数据集分割 (train 或 val)
        iou_threshold: IoU聚类阈值
        quality_threshold: 质量分数阈值
        conf_threshold: 预测置信度阈值
    """
    print("=" * 70)
    print("标签质量分析流程 (基于SafeDNN-Clean)")
    print("=" * 70)
    
    # 检查SafeDNN-Clean脚本是否存在
    safednn_script = Path("otherwork/safednn-clean/safednn-clean.py")
    if not safednn_script.exists():
        print(f"\n❌ 错误: SafeDNN-Clean脚本不存在: {safednn_script}")
        print("   请确保 otherwork/safednn-clean/safednn-clean.py 存在")
        return False
    
    # 文件路径
    annotations_file = f"annotations_{split}_coco.json"
    predictions_file = f"predictions_{split}_coco.json"  # 根据split生成不同的预测文件
    quality_report_file = f"quality_report_{split}.json"  # 根据split生成不同的报告文件
    
    # 步骤1: 转换标注为COCO格式
    print(f"\n[1/4] 转换{split}集标注为COCO格式...")
    try:
        from dataset.yolo_to_coco import yolo_to_coco
        yolo_to_coco(split, annotations_file)
    except Exception as e:
        print(f"❌ 错误: 转换标注失败: {e}")
        return False
    
    # 步骤2: 生成预测结果（COCO格式）
    print(f"\n[2/4] 生成模型预测结果...")
    try:
        from dataset.generate_predictions_coco import generate_predictions_coco
        generate_predictions_coco(
            model_weights,
            split,
            annotations_file,
            predictions_file,
            conf_threshold
        )
    except Exception as e:
        print(f"❌ 错误: 生成预测失败: {e}")
        return False
    
    # 步骤3: 运行SafeDNN-Clean分析
    print(f"\n[3/4] 运行SafeDNN-Clean分析...")
    print(f"   IoU阈值: {iou_threshold}")
    print(f"   质量阈值: {quality_threshold}")
    
    try:
        result = subprocess.run([
            sys.executable,
            str(safednn_script),
            "--iou", str(iou_threshold),
            "--threshold", str(quality_threshold),
            "-o", quality_report_file,
            annotations_file,
            predictions_file
        ], capture_output=True, text=True, check=True)
        
        print("✅ SafeDNN-Clean分析完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ 错误: SafeDNN-Clean运行失败")
        print(f"   返回码: {e.returncode}")
        if e.stdout:
            print(f"   输出: {e.stdout}")
        if e.stderr:
            print(f"   错误: {e.stderr}")
        return False
    
    # 步骤4: 分析结果
    print(f"\n[4/4] 分析结果...")
    try:
        with open(quality_report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        # 统计错误类型
        issues = {
            "spurious": 0,    # 虚假标注
            "missing": 0,     # 缺失标注
            "location": 0,    # 定位错误
            "label": 0       # 类别错误
        }
        
        quality_scores = []
        for ann in report["annotations"]:
            if "issue" in ann:
                issue_type = ann["issue"]
                if issue_type in issues:
                    issues[issue_type] += 1
                if "quality" in ann:
                    quality_scores.append(ann["quality"])
        
        # 打印摘要
        print("\n" + "=" * 70)
        print("分析结果摘要")
        print("=" * 70)
        print(f"  总标注数: {len(report['annotations'])}")
        print(f"  发现问题: {sum(issues.values())}")
        print(f"\n  错误类型分布:")
        print(f"    虚假标注 (spurious): {issues['spurious']}")
        print(f"       → 标注了但模型没检测到，可能是误标注")
        print(f"    缺失标注 (missing): {issues['missing']}")
        print(f"       → 模型检测到了但没标注，需要添加标注")
        print(f"    定位错误 (location): {issues['location']}")
        print(f"       → 类别对但边界框位置不准")
        print(f"    类别错误 (label): {issues['label']}")
        print(f"       → 检测到了但类别标注错误")
        
        if quality_scores:
            print(f"\n  质量分数统计:")
            print(f"    最低: {min(quality_scores):.3f}")
            print(f"    最高: {max(quality_scores):.3f}")
            print(f"    平均: {sum(quality_scores)/len(quality_scores):.3f}")
            print(f"    中位数: {sorted(quality_scores)[len(quality_scores)//2]:.3f}")
        
        print(f"\n  详细报告: {quality_report_file}")
        print("=" * 70)
        
        # 生成修复建议
        print("\n💡 修复建议:")
        print("   1. 运行可视化脚本查看问题:")
        print(f"      python visualize_quality_report.py")
        print("   2. 按质量分数排序，优先修复低质量标注")
        print("   3. 根据issue类型采取相应修复策略:")
        print("      - spurious: 删除误标注")
        print("      - missing: 添加缺失标注")
        print("      - location: 调整边界框位置")
        print("      - label: 修正类别")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误: 分析结果失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="运行完整的标签质量分析流程"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型权重路径"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="val",
        help="数据集分割 (train 或 val)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU聚类阈值 (默认: 0.5)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="质量分数阈值 (默认: 0.5)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="预测置信度阈值 (默认: 0.25)"
    )
    
    args = parser.parse_args()
    
    success = run_analysis_pipeline(
        args.model,
        args.split,
        args.iou,
        args.threshold,
        args.conf
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

