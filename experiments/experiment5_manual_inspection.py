#!/usr/bin/env python3
"""
实验5: Manual Inspection（人工检查实验）
运行CLOD分析，统计质量分数分布，识别需要人工检查的标注

用法:
    python experiments/experiment5_manual_inspection.py
"""

import json
import time
import sys
import subprocess
import psutil
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import argparse

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.yolo_to_coco import yolo_to_coco
from dataset.generate_predictions_coco import generate_predictions_coco

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ================================
# 配置
# ================================
WORKSPACE_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = WORKSPACE_ROOT / "experiments" / "experiment5_results"
SAFEDNN_SCRIPT = WORKSPACE_ROOT / "otherwork" / "safednn-clean" / "safednn-clean.py"

# 默认模型（如果未提供）
DEFAULT_MODEL = "runs/detect/yolov8n_baseline_new/weights/best.pt"

# 质量阈值（用于识别需要人工检查的标注）
QUALITY_CUTOFF = 0.039  # 3.9%


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def get_memory_usage():
    """获取当前内存使用量（GB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)  # 转换为GB


def run_clod_analysis(
    model_weights: str,
    split: str = "train",
    iou_threshold: float = 0.5,
    quality_threshold: float = 0.5
) -> tuple:
    """
    运行CLOD分析
    
    Returns:
        (quality_report_file, predictions_file, processing_time, memory_usage)
    """
    print_section(f"步骤1: 运行CLOD分析 ({split}集)")
    
    # 检查SafeDNN-Clean脚本
    if not SAFEDNN_SCRIPT.exists():
        raise FileNotFoundError(f"SafeDNN-Clean脚本不存在: {SAFEDNN_SCRIPT}")
    
    # 文件路径
    annotations_file = f"annotations_{split}_coco.json"
    predictions_file = f"predictions_{split}_coco.json"
    quality_report_file = f"quality_report_{split}.json"
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # 步骤1: 转换标注为COCO格式
    print(f"\n[1/4] 转换{split}集标注为COCO格式...")
    try:
        # 构建完整的数据集目录路径
        dataset_dir = WORKSPACE_ROOT / "cotton weed dataset" / split
        yolo_to_coco(str(dataset_dir), annotations_file)
    except Exception as e:
        raise RuntimeError(f"转换标注失败: {e}")
    
    # 步骤2: 生成预测结果
    print(f"\n[2/4] 生成模型预测结果...")
    try:
        generate_predictions_coco(
            model_weights,
            split,
            annotations_file,
            predictions_file,
            conf_threshold=0.25
        )
    except Exception as e:
        raise RuntimeError(f"生成预测失败: {e}")
    
    # 步骤3: 运行SafeDNN-Clean分析
    print(f"\n[3/4] 运行SafeDNN-Clean分析...")
    print(f"   IoU阈值: {iou_threshold}")
    print(f"   质量阈值: {quality_threshold}")
    
    try:
        result = subprocess.run([
            sys.executable,
            str(SAFEDNN_SCRIPT),
            "--iou", str(iou_threshold),
            "--threshold", str(quality_threshold),
            "-o", quality_report_file,
            annotations_file,
            predictions_file
        ], capture_output=True, text=True, check=True)
        
        print("✅ SafeDNN-Clean分析完成")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"SafeDNN-Clean运行失败: {e.stderr}")
    
    # 计算处理时间和内存
    processing_time = time.time() - start_time
    end_memory = get_memory_usage()
    peak_memory = end_memory - start_memory
    
    print(f"\n[4/4] 分析完成")
    print(f"   处理时间: {processing_time:.2f} 秒")
    print(f"   内存使用: {peak_memory:.2f} GiB")
    
    return quality_report_file, predictions_file, processing_time, peak_memory


def analyze_quality_distribution(quality_report_file: str) -> dict:
    """分析质量分数分布"""
    print_section("步骤2: 分析质量分数分布")
    
    with open(quality_report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    # 收集质量分数
    quality_scores = []
    issues_by_type = defaultdict(int)
    annotations_by_issue = defaultdict(list)
    
    for ann in report["annotations"]:
        if "quality" in ann:
            quality_scores.append(ann["quality"])
        
        if "issue" in ann:
            issue_type = ann["issue"]
            issues_by_type[issue_type] += 1
            annotations_by_issue[issue_type].append(ann)
    
    # 统计需要人工检查的标注
    suspicious_annotations = [
        ann for ann in report["annotations"]
        if ann.get("quality", 1.0) < QUALITY_CUTOFF
    ]
    
    # 计算统计信息
    stats = {
        "total_annotations": len(report["annotations"]),
        "annotations_with_quality": len(quality_scores),
        "quality_scores": quality_scores,
        "issues_by_type": dict(issues_by_type),
        "suspicious_count": len(suspicious_annotations),
        "suspicious_percentage": len(suspicious_annotations) / len(report["annotations"]) * 100 if report["annotations"] else 0,
        "quality_stats": {}
    }
    
    if quality_scores:
        stats["quality_stats"] = {
            "min": float(np.min(quality_scores)),
            "max": float(np.max(quality_scores)),
            "mean": float(np.mean(quality_scores)),
            "median": float(np.median(quality_scores)),
            "std": float(np.std(quality_scores)),
            "percentile_10": float(np.percentile(quality_scores, 10)),
            "percentile_25": float(np.percentile(quality_scores, 25)),
            "percentile_75": float(np.percentile(quality_scores, 75)),
            "percentile_90": float(np.percentile(quality_scores, 90))
        }
    
    # 打印统计信息
    print(f"\n总标注数: {stats['total_annotations']}")
    print(f"有质量分数的标注: {stats['annotations_with_quality']}")
    
    if stats["quality_stats"]:
        print(f"\n质量分数统计:")
        print(f"  最低: {stats['quality_stats']['min']:.4f}")
        print(f"  最高: {stats['quality_stats']['max']:.4f}")
        print(f"  平均: {stats['quality_stats']['mean']:.4f}")
        print(f"  中位数: {stats['quality_stats']['median']:.4f}")
        print(f"  标准差: {stats['quality_stats']['std']:.4f}")
        print(f"  10%分位数: {stats['quality_stats']['percentile_10']:.4f}")
        print(f"  25%分位数: {stats['quality_stats']['percentile_25']:.4f}")
        print(f"  75%分位数: {stats['quality_stats']['percentile_75']:.4f}")
        print(f"  90%分位数: {stats['quality_stats']['percentile_90']:.4f}")
    
    print(f"\n问题类型分布:")
    for issue_type, count in stats["issues_by_type"].items():
        print(f"  {issue_type}: {count}")
    
    print(f"\n需要人工检查的标注 (质量分数 < {QUALITY_CUTOFF}):")
    print(f"  数量: {stats['suspicious_count']}")
    print(f"  比例: {stats['suspicious_percentage']:.2f}%")
    
    return stats


def plot_quality_distribution(quality_scores: list, output_file: str):
    """绘制质量分数分布图"""
    print_section("步骤3: 绘制质量分数分布图")
    
    if not quality_scores:
        print("⚠️  没有质量分数数据，跳过绘图")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 直方图
    ax = axes[0, 0]
    ax.hist(quality_scores, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(QUALITY_CUTOFF, color='r', linestyle='--', linewidth=2, label=f'Cutoff ({QUALITY_CUTOFF})')
    ax.set_xlabel('Quality Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Quality Score Distribution', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 累积分布
    ax = axes[0, 1]
    sorted_scores = np.sort(quality_scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax.plot(sorted_scores, cumulative, linewidth=2)
    ax.axvline(QUALITY_CUTOFF, color='r', linestyle='--', linewidth=2, label=f'Cutoff ({QUALITY_CUTOFF})')
    ax.set_xlabel('Quality Score', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 箱线图
    ax = axes[1, 0]
    ax.boxplot(quality_scores, vert=True)
    ax.axhline(QUALITY_CUTOFF, color='r', linestyle='--', linewidth=2, label=f'Cutoff ({QUALITY_CUTOFF})')
    ax.set_ylabel('Quality Score', fontsize=12)
    ax.set_title('Quality Score Box Plot', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 对数尺度直方图（更好地显示低质量分数）
    ax = axes[1, 1]
    log_scores = [-np.log10(max(score, 1e-6)) for score in quality_scores]
    ax.hist(log_scores, bins=50, edgecolor='black', alpha=0.7)
    cutoff_log = -np.log10(QUALITY_CUTOFF)
    ax.axvline(cutoff_log, color='r', linestyle='--', linewidth=2, label=f'Cutoff ({QUALITY_CUTOFF})')
    ax.set_xlabel('-log10(Quality Score)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Quality Score Distribution (Log Scale)', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_file}")


def generate_report(
    stats: dict,
    processing_time: float,
    memory_usage: float,
    quality_report_file: str,
    output_file: str
):
    """生成完整的实验报告"""
    print_section("步骤4: 生成实验报告")
    
    # 计算处理速度
    annotations_per_second = stats["total_annotations"] / processing_time if processing_time > 0 else 0
    memory_per_annotation = (memory_usage * 1024) / stats["total_annotations"] if stats["total_annotations"] > 0 else 0  # KiB per annotation
    
    report = {
        "experiment": "Manual Inspection",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quality_report_file": quality_report_file,
        "processing": {
            "time_seconds": processing_time,
            "memory_gib": memory_usage,
            "annotations_per_second": annotations_per_second,
            "memory_per_annotation_kib": memory_per_annotation
        },
        "statistics": stats,
        "quality_cutoff": QUALITY_CUTOFF,
        "recommendations": {
            "suspicious_annotations_to_review": stats["suspicious_count"],
            "suspicious_percentage": stats["suspicious_percentage"]
        }
    }
    
    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 报告已保存: {output_file}")
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("实验摘要")
    print("=" * 70)
    print(f"处理时间: {processing_time:.2f} 秒")
    print(f"处理速度: {annotations_per_second:.1f} 标注/秒")
    print(f"内存使用: {memory_usage:.2f} GiB")
    print(f"每标注内存: {memory_per_annotation:.2f} KiB")
    print(f"\n需要人工检查的标注: {stats['suspicious_count']} ({stats['suspicious_percentage']:.2f}%)")
    print("=" * 70)
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="实验5: Manual Inspection（人工检查实验）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"模型权重路径 (默认: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="train",
        help="数据集分割 (默认: train)"
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
    
    args = parser.parse_args()
    
    # 检查模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        print(f"   请先训练baseline模型或提供正确的模型路径")
        return
    
    # 创建输出目录
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("实验5: Manual Inspection（人工检查实验）")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"数据集: {args.split}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print("=" * 70)
    
    try:
        # 步骤1: 运行CLOD分析
        quality_report_file, predictions_file, processing_time, memory_usage = run_clod_analysis(
            args.model,
            args.split,
            args.iou,
            args.threshold
        )
        
        # 步骤2: 分析质量分布
        stats = analyze_quality_distribution(quality_report_file)
        
        # 步骤3: 绘制分布图
        if stats["quality_scores"]:
            plot_file = OUTPUT_ROOT / "quality_distribution.png"
            plot_quality_distribution(stats["quality_scores"], str(plot_file))
        
        # 步骤4: 生成报告
        report_file = OUTPUT_ROOT / "manual_inspection_report.json"
        report = generate_report(
            stats,
            processing_time,
            memory_usage,
            quality_report_file,
            str(report_file)
        )
        
        print("\n" + "=" * 70)
        print("✅ 实验完成!")
        print("=" * 70)
        print(f"报告文件: {report_file}")
        if stats["quality_scores"]:
            print(f"分布图: {OUTPUT_ROOT / 'quality_distribution.png'}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

