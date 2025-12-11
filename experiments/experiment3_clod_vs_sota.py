#!/usr/bin/env python3
"""
实验3: CLOD vs State-of-the-Art（CLOD与SOTA方法对比实验）
对比CLOD和ObjectLab在检测人工噪声上的性能

注意: ObjectLab需要单独实现或安装，当前只实现CLOD部分

用法:
    python experiments/experiment3_clod_vs_sota.py
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
OUTPUT_ROOT = WORKSPACE_ROOT / "experiments" / "experiment3_results"
SAFEDNN_SCRIPT = WORKSPACE_ROOT / "otherwork" / "safednn-clean" / "safednn-clean.py"

# 默认模型
DEFAULT_MODEL = "runs/detect/yolov8n_baseline_new/weights/best.pt"

# 噪声配置
NOISE_RATIO = 0.25  # 25%噪声（论文中的设置）
NOISE_TYPES = ["label", "location", "scale", "spurious", "missing"]
LOCATION_MAGNITUDES = [0.2, 0.5]
SCALE_MAGNITUDES = [0.2, 0.5]

# IoU阈值（用于CLOD）
CLOD_IOU_THRESHOLD = 0.5


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def calculate_auroc_clod(
    quality_report_file: str,
    ground_truth_noisy_ids: set,
    noise_type: str
) -> float:
    """计算CLOD的AUROC"""
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


def run_objectlab(
    noisy_annotations_file: str,
    predictions_file: str
) -> dict:
    """
    运行ObjectLab方法
    
    注意: 这是一个占位函数，需要实现ObjectLab或使用替代方法
    如果ObjectLab不可用，返回None
    
    Returns:
        ObjectLab的结果字典，包含质量分数等
    """
    # TODO: 实现ObjectLab
    # 这里返回None表示ObjectLab不可用
    print("⚠️  ObjectLab未实现，跳过ObjectLab分析")
    return None


def calculate_auroc_objectlab(
    objectlab_results: dict,
    ground_truth_noisy_ids: set,
    noise_type: str
) -> float:
    """计算ObjectLab的AUROC"""
    if objectlab_results is None:
        return None
    
    # TODO: 实现ObjectLab的AUROC计算
    return None


def evaluate_clod_vs_sota(
    model_weights: str,
    split: str = "val",
    noise_ratio: float = 0.25
) -> dict:
    """评估CLOD vs SOTA方法"""
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
    
    print_section("步骤2: 测试不同噪声类型")
    
    # 对每种噪声类型
    for noise_type in NOISE_TYPES:
        print(f"\n{'='*70}")
        print(f"测试噪声类型: {noise_type}")
        print(f"{'='*70}")
        
        if noise_type == "location":
            magnitudes = LOCATION_MAGNITUDES
        elif noise_type == "scale":
            magnitudes = SCALE_MAGNITUDES
        else:
            magnitudes = [None]
        
        for magnitude in magnitudes:
            # 创建噪声数据集
            if magnitude is not None:
                noise_name = f"{noise_type}_{int(magnitude*100)}"
            else:
                noise_name = noise_type
            
            noisy_file = f"annotations_{split}_noisy_{noise_name}.json"
            
            print(f"\n生成噪声数据集: {noise_name}")
            
            # 添加噪声
            if magnitude is not None:
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
            print(f"  噪声标注数: {len(ground_truth_noisy_ids)}")
            
            # 运行CLOD
            print(f"\n  运行CLOD...")
            quality_report_file = f"quality_report_{noise_name}.json"
            
            try:
                subprocess.run([
                    sys.executable,
                    str(SAFEDNN_SCRIPT),
                    "--iou", str(CLOD_IOU_THRESHOLD),
                    "--threshold", "0.5",
                    "-o", quality_report_file,
                    noisy_file,
                    predictions_file
                ], capture_output=True, text=True, check=True)
                
                clod_auroc = calculate_auroc_clod(
                    quality_report_file,
                    ground_truth_noisy_ids,
                    noise_type
                )
                print(f"    CLOD AUROC: {clod_auroc:.4f}")
            except subprocess.CalledProcessError as e:
                print(f"    ❌ CLOD分析失败: {e.stderr}")
                clod_auroc = 0.0
            
            # 运行ObjectLab（如果可用）
            print(f"\n  运行ObjectLab...")
            objectlab_results = run_objectlab(noisy_file, predictions_file)
            
            if objectlab_results is not None:
                objectlab_auroc = calculate_auroc_objectlab(
                    objectlab_results,
                    ground_truth_noisy_ids,
                    noise_type
                )
                print(f"    ObjectLab AUROC: {objectlab_auroc:.4f}")
            else:
                objectlab_auroc = None
            
            # 保存结果
            results[noise_name] = {
                "noise_type": noise_type,
                "noise_ratio": noise_ratio,
                "magnitude": magnitude,
                "clod_auroc": clod_auroc,
                "objectlab_auroc": objectlab_auroc,
                "num_noisy": len(ground_truth_noisy_ids)
            }
    
    return results


def plot_comparison_results(results: dict, output_file: str):
    """绘制对比结果图"""
    print_section("步骤3: 绘制对比结果图")
    
    # 组织数据
    noise_types = []
    clod_aurocs = []
    objectlab_aurocs = []
    
    for noise_name, result in results.items():
        noise_types.append(noise_name)
        clod_aurocs.append(result["clod_auroc"])
        if result["objectlab_auroc"] is not None:
            objectlab_aurocs.append(result["objectlab_auroc"])
        else:
            objectlab_aurocs.append(None)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(noise_types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, clod_aurocs, width, label='CLOD', alpha=0.8, color='#2ca02c')
    
    # 只绘制ObjectLab有值的部分
    valid_objectlab = [i for i, v in enumerate(objectlab_aurocs) if v is not None]
    if valid_objectlab:
        objectlab_values = [objectlab_aurocs[i] for i in valid_objectlab]
        bars2 = ax.bar([x[i] + width/2 for i in valid_objectlab], objectlab_values, width, 
                      label='ObjectLab', alpha=0.8, color='#1f77b4')
    
    # 添加数值标签
    for i, (clod, obj) in enumerate(zip(clod_aurocs, objectlab_aurocs)):
        ax.text(i - width/2, clod + 0.01, f'{clod:.3f}', ha='center', va='bottom', fontsize=9)
        if obj is not None:
            ax.text(i + width/2, obj + 0.01, f'{obj:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Noise Type', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('CLOD vs ObjectLab: AUROC Comparison', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(noise_types, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_file}")


def generate_comparison_table(results: dict) -> str:
    """生成对比表格（Markdown格式）"""
    print_section("步骤4: 生成对比表格")
    
    table_lines = [
        "| Noise Type | Dataset | CLOD | ObjectLab |",
        "|------------|---------|------|-----------|"
    ]
    
    for noise_name, result in results.items():
        noise_type = result["noise_type"]
        clod_auroc = result["clod_auroc"]
        objectlab_auroc = result["objectlab_auroc"]
        
        clod_str = f"{clod_auroc:.3f}" if clod_auroc else "N/A"
        obj_str = f"{objectlab_auroc:.3f}" if objectlab_auroc else "N/A"
        
        table_lines.append(f"| {noise_type} | {noise_name} | {clod_str} | {obj_str} |")
    
    table = "\n".join(table_lines)
    print("\n对比表格:")
    print(table)
    
    return table


def main():
    parser = argparse.ArgumentParser(
        description="实验3: CLOD vs State-of-the-Art（CLOD与SOTA方法对比实验）"
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
        default=0.25,
        help="噪声比例 (默认: 0.25)"
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
    print("实验3: CLOD vs State-of-the-Art（CLOD与SOTA方法对比实验）")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"数据集: {args.split}")
    print(f"噪声比例: {args.noise_ratio * 100:.0f}%")
    print(f"输出目录: {OUTPUT_ROOT}")
    print("=" * 70)
    print("\n⚠️  注意: ObjectLab需要单独实现，当前只运行CLOD部分")
    print("=" * 70)
    
    try:
        # 评估CLOD vs SOTA
        results = evaluate_clod_vs_sota(
            args.model,
            args.split,
            args.noise_ratio
        )
        
        # 保存结果
        results_file = OUTPUT_ROOT / "clod_vs_sota_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 结果已保存: {results_file}")
        
        # 绘制对比图
        plot_file = OUTPUT_ROOT / "clod_vs_sota_comparison.png"
        plot_comparison_results(results, str(plot_file))
        
        # 生成对比表格
        table = generate_comparison_table(results)
        table_file = OUTPUT_ROOT / "comparison_table.md"
        with open(table_file, 'w', encoding='utf-8') as f:
            f.write("# CLOD vs ObjectLab Comparison\n\n")
            f.write(table)
        print(f"\n✅ 对比表格已保存: {table_file}")
        
        # 生成完整报告
        report = {
            "experiment": "CLOD vs State-of-the-Art",
            "timestamp": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note": "ObjectLab未实现，只包含CLOD结果",
            "results": results
        }
        
        report_file = OUTPUT_ROOT / "clod_vs_sota_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 完整报告已保存: {report_file}")
        
        # 打印摘要
        print("\n" + "=" * 70)
        print("实验摘要")
        print("=" * 70)
        for noise_name, result in results.items():
            print(f"{noise_name}:")
            print(f"  CLOD AUROC: {result['clod_auroc']:.4f}")
            if result['objectlab_auroc'] is not None:
                print(f"  ObjectLab AUROC: {result['objectlab_auroc']:.4f}")
            else:
                print(f"  ObjectLab AUROC: N/A (未实现)")
        print("=" * 70)
        
        print("\n✅ 实验完成!")
        print(f"结果文件: {results_file}")
        print(f"报告文件: {report_file}")
        print(f"图表文件: {plot_file}")
        print(f"表格文件: {table_file}")
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

