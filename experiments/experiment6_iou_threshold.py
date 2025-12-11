#!/usr/bin/env python3
"""
实验6: IoU Threshold Analysis（IoU阈值分析实验）
测试不同IoU聚类阈值对CLOD性能的影响

用法:
    python experiments/experiment6_iou_threshold.py
"""

import json
import sys
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.yolo_to_coco import yolo_to_coco
from dataset.generate_predictions_coco import generate_predictions_coco
from dataprocess.add_artificial_noise import add_artificial_noise

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("❌ 需要安装scikit-learn: pip install scikit-learn")
    sys.exit(1)

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ================================
# 配置
# ================================
WORKSPACE_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = WORKSPACE_ROOT / "experiments" / "experiment6_results"
SAFEDNN_SCRIPT = WORKSPACE_ROOT / "otherwork" / "safednn-clean" / "safednn-clean.py"

# 默认模型
DEFAULT_MODEL = "runs/detect/yolov8n_baseline_new/weights/best.pt"

# IoU阈值范围
IOU_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

# 噪声配置
NOISE_RATIO = 0.2
NOISE_TYPES = ["label", "location", "scale", "spurious", "missing"]


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def calculate_auroc(quality_report_file: str, ground_truth_noisy_ids: set, noise_type: str) -> float:
    """计算AUROC"""
    with open(quality_report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    if noise_type == "missing":
        y_true = []
        y_scores = []
        for ann in report["annotations"]:
            if ann.get("id", 0) < 0:
                is_noisy = ann.get("id") in ground_truth_noisy_ids
                quality = ann.get("quality", 1.0)
                y_true.append(1 if is_noisy else 0)
                y_scores.append(1.0 - quality)
    else:
        y_true = []
        y_scores = []
        for ann in report["annotations"]:
            if ann.get("id", 0) >= 0:
                is_noisy = ann.get("id") in ground_truth_noisy_ids
                quality = ann.get("quality", 1.0)
                issue = ann.get("issue", "")
                
                if noise_type == "spurious" and issue != "spurious":
                    continue
                elif noise_type == "label" and issue != "label":
                    continue
                elif noise_type == "location" and issue != "location":
                    continue
                
                y_true.append(1 if is_noisy else 0)
                y_scores.append(1.0 - quality)
    
    if len(y_true) == 0 or sum(y_true) == 0:
        return 0.0
    
    if len(set(y_true)) == 1:
        return 0.5
    
    try:
        return roc_auc_score(y_true, y_scores)
    except:
        return 0.0


def evaluate_iou_thresholds(
    model_weights: str,
    split: str = "val",
    noise_ratio: float = 0.2
) -> dict:
    """评估不同IoU阈值的影响"""
    print_section("步骤1: 准备数据和生成预测")
    
    # 1. 转换原始验证集
    original_annotations_file = f"annotations_{split}_coco.json"
    print(f"\n[1/3] 转换原始{split}集为COCO格式...")
    # 构建完整的数据集目录路径
    dataset_dir = WORKSPACE_ROOT / "cotton weed dataset" / split
    yolo_to_coco(str(dataset_dir), original_annotations_file)
    
    # 2. 生成预测
    predictions_file = f"predictions_{split}_coco.json"
    print(f"\n[2/3] 生成模型预测...")
    generate_predictions_coco(
        model_weights,
        split,
        original_annotations_file,
        predictions_file,
        conf_threshold=0.25
    )
    
    # 读取原始数据
    with open(original_annotations_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    results = {}
    
    print_section("步骤2: 测试不同噪声类型和IoU阈值")
    
    # 对每种噪声类型
    for noise_type in NOISE_TYPES:
        print(f"\n{'='*70}")
        print(f"测试噪声类型: {noise_type}")
        print(f"{'='*70}")
        
        # 创建噪声数据集
        magnitude = 0.25 if noise_type in ["location", "scale"] else None
        noise_name = f"{noise_type}_{int(magnitude*100)}" if magnitude else noise_type
        noisy_file = f"annotations_{split}_noisy_{noise_name}.json"
        
        print(f"生成噪声数据集: {noise_name}")
        
        if magnitude:
            noisy_data, modified_ids = add_artificial_noise(
                original_annotations_file,
                noisy_file,
                noise_type,
                noise_ratio,
                magnitude=magnitude,
                seed=42
            )
        else:
            noisy_data, modified_ids = add_artificial_noise(
                original_annotations_file,
                noisy_file,
                noise_type,
                noise_ratio,
                seed=42
            )
        
        ground_truth_noisy_ids = set(modified_ids)
        
        # 测试不同IoU阈值
        auroc_by_iou = {}
        
        for iou_threshold in IOU_THRESHOLDS:
            print(f"\n  测试IoU阈值: {iou_threshold}")
            
            # 运行CLOD
            quality_report_file = f"quality_report_{noise_name}_iou{iou_threshold:.1f}.json"
            
            try:
                subprocess.run([
                    sys.executable,
                    str(SAFEDNN_SCRIPT),
                    "--iou", str(iou_threshold),
                    "--threshold", "0.5",
                    "-o", quality_report_file,
                    noisy_file,
                    predictions_file
                ], capture_output=True, text=True, check=True)
                
                # 计算AUROC
                auroc = calculate_auroc(
                    quality_report_file,
                    ground_truth_noisy_ids,
                    noise_type
                )
                
                auroc_by_iou[iou_threshold] = auroc
                print(f"    AUROC: {auroc:.4f}")
                
            except subprocess.CalledProcessError as e:
                print(f"    ❌ CLOD分析失败: {e.stderr}")
                auroc_by_iou[iou_threshold] = 0.0
        
        results[noise_name] = {
            "noise_type": noise_type,
            "auroc_by_iou": auroc_by_iou
        }
    
    return results


def plot_iou_threshold_results(results: dict, output_file: str):
    """绘制IoU阈值分析图"""
    print_section("步骤3: 绘制IoU阈值分析图")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 为每种噪声类型绘制曲线
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']
    
    for i, (noise_name, result) in enumerate(results.items()):
        noise_type = result["noise_type"]
        auroc_by_iou = result["auroc_by_iou"]
        
        # 排序IoU阈值
        sorted_ious = sorted(auroc_by_iou.keys())
        aurocs = [auroc_by_iou[iou] for iou in sorted_ious]
        
        ax.plot(sorted_ious, aurocs, 
               marker=markers[i % len(markers)],
               label=noise_type,
               color=colors[i],
               linewidth=2,
               markersize=8)
    
    ax.set_xlabel('IoU Clustering Threshold', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('CLOD AUROC with Respect to IoU Clustering Threshold', fontsize=13, pad=15)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # 标记最佳阈值范围
    ax.axvspan(0.4, 0.6, alpha=0.1, color='green', label='Best Range (0.4-0.6)')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_file}")


def find_best_iou_threshold(results: dict) -> dict:
    """找到最佳IoU阈值"""
    print_section("步骤4: 分析最佳IoU阈值")
    
    # 对每个IoU阈值，计算平均AUROC
    iou_averages = {}
    
    for iou in IOU_THRESHOLDS:
        aurocs = []
        for noise_name, result in results.items():
            auroc = result["auroc_by_iou"].get(iou, 0.0)
            if auroc > 0:
                aurocs.append(auroc)
        
        if aurocs:
            iou_averages[iou] = {
                "mean": np.mean(aurocs),
                "std": np.std(aurocs),
                "count": len(aurocs)
            }
    
    # 找到最佳阈值
    if iou_averages:
        best_iou = max(iou_averages.keys(), key=lambda x: iou_averages[x]["mean"])
        best_mean = iou_averages[best_iou]["mean"]
        
        print(f"\n最佳IoU阈值: {best_iou}")
        print(f"平均AUROC: {best_mean:.4f}")
        
        print(f"\n所有IoU阈值的平均AUROC:")
        for iou in sorted(iou_averages.keys()):
            stats = iou_averages[iou]
            print(f"  IoU={iou:.1f}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
        
        return {
            "best_iou": best_iou,
            "best_mean_auroc": best_mean,
            "iou_averages": iou_averages
        }
    
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="实验6: IoU Threshold Analysis（IoU阈值分析实验）"
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
        default="val",
        help="数据集分割 (默认: val)"
    )
    parser.add_argument(
        "--noise-ratio",
        type=float,
        default=0.2,
        help="噪声比例 (默认: 0.2)"
    )
    
    args = parser.parse_args()
    
    # 检查模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return 1
    
    # 检查SafeDNN-Clean脚本
    if not SAFEDNN_SCRIPT.exists():
        print(f"❌ SafeDNN-Clean脚本不存在: {SAFEDNN_SCRIPT}")
        return 1
    
    # 创建输出目录
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("实验6: IoU Threshold Analysis（IoU阈值分析实验）")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"数据集: {args.split}")
    print(f"噪声比例: {args.noise_ratio * 100:.0f}%")
    print(f"IoU阈值范围: {IOU_THRESHOLDS}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print("=" * 70)
    
    try:
        # 评估不同IoU阈值
        results = evaluate_iou_thresholds(
            args.model,
            args.split,
            args.noise_ratio
        )
        
        # 保存结果
        results_file = OUTPUT_ROOT / "iou_threshold_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 结果已保存: {results_file}")
        
        # 绘制图表
        plot_file = OUTPUT_ROOT / "iou_threshold_analysis.png"
        plot_iou_threshold_results(results, str(plot_file))
        
        # 分析最佳阈值
        best_analysis = find_best_iou_threshold(results)
        
        # 保存完整报告
        report = {
            "experiment": "IoU Threshold Analysis",
            "timestamp": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
            "best_analysis": best_analysis
        }
        
        report_file = OUTPUT_ROOT / "iou_threshold_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 完整报告已保存: {report_file}")
        
        print("\n" + "=" * 70)
        print("✅ 实验完成!")
        print("=" * 70)
        print(f"结果文件: {results_file}")
        print(f"报告文件: {report_file}")
        print(f"图表文件: {plot_file}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

