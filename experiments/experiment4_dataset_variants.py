#!/usr/bin/env python3
"""
实验4: Dataset Variants（数据集变体实验）
创建不同的数据集变体，训练模型并对比性能

数据集变体：
1. Original: 原始数据集
2. Suggestions: 应用CLOD建议的前20%
3. Selected: 匹配人工修订的变体（如果有）

用法:
    python experiments/experiment4_dataset_variants.py
"""

import json
import sys
import subprocess
import shutil
import yaml
from pathlib import Path
from collections import defaultdict
import pandas as pd
import argparse

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.yolo_to_coco import yolo_to_coco
from dataset.coco_to_yolo import coco_to_yolo
from dataset.generate_predictions_coco import generate_predictions_coco
from tools.clean_dataset import clean_dataset
from ultralytics import YOLO

# ================================
# 配置
# ================================
WORKSPACE_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = WORKSPACE_ROOT / "experiments" / "experiment4_results"
SAFEDNN_SCRIPT = WORKSPACE_ROOT / "otherwork" / "safednn-clean" / "safednn-clean.py"

# 默认模型
DEFAULT_MODEL = "runs/detect/yolov8n_baseline_new/weights/best.pt"
MODEL_WEIGHTS = "yolov8n.pt"

# 训练配置
EPOCHS = 3
BATCH_SIZE = 8
IMAGE_SIZE = 640
DEVICE = 0

# 清洗阈值
LOCATION_THRESHOLD = 0.7
LABEL_THRESHOLD = 0.8
MISSING_THRESHOLD = 0.5

# Suggestions比例（前20%）
SUGGESTIONS_RATIO = 0.2


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def train_model(dataset_yaml: str, run_name: str) -> dict:
    """训练模型并返回性能指标"""
    print(f"\n训练模型: {run_name}")
    print(f"  数据集: {dataset_yaml}")
    
    model = YOLO(MODEL_WEIGHTS)
    
    try:
        results = model.train(
            data=str(dataset_yaml),
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            workers=0,
            project=str(OUTPUT_ROOT / "runs"),
            name=run_name,
            val=True,
            save=True,
            plots=True
        )
        
        # 提取性能指标
        metrics = {}
        
        # 尝试从results获取
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict.copy()
        
        # 如果无法从results获取，尝试从results.csv读取
        results_csv = OUTPUT_ROOT / "runs" / run_name / "results.csv"
        if results_csv.exists():
            try:
                df = pd.read_csv(results_csv)
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    for col in df.columns:
                        if 'map' in col.lower():
                            metrics[col] = float(last_row[col])
            except Exception as e:
                print(f"警告: 读取results.csv失败: {e}")
        
        # 提取主要指标
        map50 = metrics.get('metrics/mAP50(B)', None)
        map75 = metrics.get('metrics/mAP75(B)', None)
        map = metrics.get('metrics/mAP50-95(B)', None)
        
        print(f"  ✅ 训练完成")
        if map50:
            print(f"    mAP@0.5: {map50:.4f}")
        if map75:
            print(f"    mAP@0.75: {map75:.4f}")
        if map:
            print(f"    mAP@0.5:0.95: {map:.4f}")
        
        return {
            "mAP50": map50,
            "mAP75": map75,
            "mAP": map,
            "all_metrics": metrics
        }
        
    except Exception as e:
        print(f"  ❌ 训练失败: {e}")
        return None


def create_suggestions_dataset(
    quality_report_file: str,
    predictions_file: str,
    original_annotations_file: str,
    output_file: str,
    top_ratio: float = 0.2
) -> dict:
    """
    创建Suggestions数据集（应用CLOD建议的前top_ratio%）
    
    Args:
        quality_report_file: CLOD质量报告
        predictions_file: 预测结果
        original_annotations_file: 原始标注文件
        output_file: 输出文件
        top_ratio: 应用前多少比例的建议（默认0.2，即20%）
    
    Returns:
        统计信息
    """
    print_section("步骤2: 创建Suggestions数据集")
    
    # 读取质量报告
    with open(quality_report_file, 'r', encoding='utf-8') as f:
        quality_report = json.load(f)
    
    # 读取原始标注
    with open(original_annotations_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # 读取预测结果
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)
    
    # 按质量分数排序所有建议（包括spurious, location, label, missing）
    all_suggestions = []
    
    for ann in quality_report["annotations"]:
        if "issue" in ann and "quality" in ann:
            suggestion = {
                "annotation": ann,
                "quality": ann["quality"],
                "issue": ann["issue"]
            }
            all_suggestions.append(suggestion)
    
    # 按质量分数排序（质量越低，优先级越高）
    all_suggestions.sort(key=lambda x: x["quality"])
    
    # 选择前top_ratio%
    num_to_apply = int(len(all_suggestions) * top_ratio)
    selected_suggestions = all_suggestions[:num_to_apply]
    
    print(f"总建议数: {len(all_suggestions)}")
    print(f"应用前 {top_ratio*100:.0f}%: {num_to_apply} 个建议")
    
    # 应用建议（使用clean_dataset函数）
    # 但我们需要修改它，只应用选中的建议
    # 为了简化，我们直接调用clean_dataset，但设置更高的阈值来只应用高质量建议
    
    # 计算质量阈值（前top_ratio%的质量分数上限）
    if selected_suggestions:
        quality_threshold = selected_suggestions[-1]["quality"]
    else:
        quality_threshold = 1.0
    
    print(f"质量分数阈值: {quality_threshold:.4f}")
    
    # 创建临时质量报告（只包含选中的建议）
    temp_quality_report = quality_report.copy()
    temp_quality_report["annotations"] = [
        ann for ann in quality_report["annotations"]
        if "quality" not in ann or ann.get("quality", 1.0) <= quality_threshold
    ]
    
    temp_quality_file = "temp_quality_report_suggestions.json"
    with open(temp_quality_file, 'w', encoding='utf-8') as f:
        json.dump(temp_quality_report, f, indent=2, ensure_ascii=False)
    
    # 应用清洗
    cleaned_data = clean_dataset(
        quality_report_file=temp_quality_file,
        predictions_file=predictions_file,
        output_file=output_file,
        location_score_threshold=LOCATION_THRESHOLD,
        label_score_threshold=LABEL_THRESHOLD,
        missing_score_threshold=MISSING_THRESHOLD
    )
    
    # 清理临时文件
    Path(temp_quality_file).unlink(missing_ok=True)
    
    # 统计信息
    stats = {
        "total_suggestions": len(all_suggestions),
        "applied_suggestions": num_to_apply,
        "quality_threshold": quality_threshold,
        "original_annotations": len(original_data["annotations"]),
        "suggestions_annotations": len(cleaned_data["annotations"])
    }
    
    return stats


def create_dataset_yaml(
    train_dir: str,
    output_yaml: str,
    val_dir: str = None
):
    """创建数据集YAML配置文件"""
    if val_dir is None:
        val_dir = "cotton weed dataset/val/images"
    
    # 处理train_dir路径
    if Path(train_dir).is_absolute():
        train_path = Path(train_dir).relative_to(WORKSPACE_ROOT) / "images"
    else:
        # 如果是相对路径，先构建完整路径
        train_full_path = WORKSPACE_ROOT / train_dir
        train_path = train_full_path.relative_to(WORKSPACE_ROOT) / "images"
    
    config = {
        "path": str(WORKSPACE_ROOT.absolute()),
        "train": str(train_path),
        "val": str(val_dir if Path(val_dir).is_absolute() else val_dir),
        "nc": 3,
        "names": {
            0: "carpetweed",
            1: "morningglory",
            2: "palmer_amaranth"
        }
    }
    
    with open(output_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return output_yaml


def prepare_yolo_dataset(
    coco_file: str,
    split: str,
    output_dir: str
):
    """准备YOLO格式数据集"""
    print(f"\n转换COCO格式到YOLO格式...")
    
    # 确定原始数据集目录
    original_dir = WORKSPACE_ROOT / "cotton weed dataset" / split
    
    # 转换标注
    coco_to_yolo(
        coco_file=coco_file,
        split_dir=str(original_dir),
        output_dir=output_dir
    )
    
    # 复制图片
    images_src = original_dir / "images"
    images_dst = Path(output_dir) / "images"
    images_dst.mkdir(parents=True, exist_ok=True)
    
    print(f"复制图片文件...")
    for img_file in images_src.glob("*.jpg"):
        shutil.copy2(img_file, images_dst / img_file.name)
    
    print(f"✅ 数据集准备完成: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="实验4: Dataset Variants（数据集变体实验）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"用于生成预测的模型权重 (默认: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="train",
        help="数据集分割 (默认: train)"
    )
    parser.add_argument(
        "--suggestions-ratio",
        type=float,
        default=0.2,
        help="Suggestions数据集应用的建议比例 (默认: 0.2)"
    )
    
    args = parser.parse_args()
    
    # 检查模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return 1
    
    # 创建输出目录
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("实验4: Dataset Variants（数据集变体实验）")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"数据集: {args.split}")
    print(f"Suggestions比例: {args.suggestions_ratio * 100:.0f}%")
    print(f"输出目录: {OUTPUT_ROOT}")
    print("=" * 70)
    
    try:
        split = args.split
        
        # 步骤1: 准备原始数据集和运行CLOD分析
        print_section("步骤1: 准备原始数据集和CLOD分析")
        
        # 转换原始数据集
        original_annotations_file = f"annotations_{split}_coco.json"
        print(f"\n[1/3] 转换原始{split}集为COCO格式...")
        # 构建完整的数据集目录路径
        dataset_dir = WORKSPACE_ROOT / "cotton weed dataset" / split
        yolo_to_coco(str(dataset_dir), original_annotations_file)
        
        # 生成预测
        predictions_file = f"predictions_{split}_coco.json"
        print(f"\n[2/3] 生成模型预测...")
        generate_predictions_coco(
            args.model,
            split,
            original_annotations_file,
            predictions_file,
            conf_threshold=0.25
        )
        
        # 运行CLOD分析
        quality_report_file = f"quality_report_{split}.json"
        print(f"\n[3/3] 运行CLOD分析...")
        subprocess.run([
            sys.executable,
            str(SAFEDNN_SCRIPT),
            "--iou", "0.5",
            "--threshold", "0.5",
            "-o", quality_report_file,
            original_annotations_file,
            predictions_file
        ], check=True)
        
        # 步骤2: 创建Suggestions数据集
        suggestions_annotations_file = f"annotations_{split}_suggestions.json"
        suggestions_stats = create_suggestions_dataset(
            quality_report_file,
            predictions_file,
            original_annotations_file,
            suggestions_annotations_file,
            top_ratio=args.suggestions_ratio
        )
        
        # 步骤3: 准备数据集并训练模型
        print_section("步骤3: 训练不同数据集变体的模型")
        
        results = {}
        
        # 3.1 原始数据集
        print(f"\n{'='*70}")
        print("训练原始数据集模型")
        print(f"{'='*70}")
        original_yaml = OUTPUT_ROOT / "dataset_original.yaml"
        create_dataset_yaml(f"{split}", str(original_yaml))
        original_perf = train_model(str(original_yaml), "original")
        if original_perf:
            results["original"] = original_perf
        
        # 3.2 Suggestions数据集
        print(f"\n{'='*70}")
        print("训练Suggestions数据集模型")
        print(f"{'='*70}")
        suggestions_dir = OUTPUT_ROOT / f"{split}_suggestions"
        prepare_yolo_dataset(suggestions_annotations_file, split, str(suggestions_dir))
        suggestions_yaml = OUTPUT_ROOT / "dataset_suggestions.yaml"
        create_dataset_yaml(str(suggestions_dir), str(suggestions_yaml))
        suggestions_perf = train_model(str(suggestions_yaml), "suggestions")
        if suggestions_perf:
            results["suggestions"] = suggestions_perf
        
        # 步骤4: 生成对比报告
        print_section("步骤4: 生成对比报告")
        
        report = {
            "experiment": "Dataset Variants",
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "split": split,
            "suggestions_ratio": args.suggestions_ratio,
            "suggestions_stats": suggestions_stats,
            "results": results
        }
        
        # 计算改进
        if "original" in results and "suggestions" in results:
            orig_map50 = results["original"].get("mAP50", 0)
            sugg_map50 = results["suggestions"].get("mAP50", 0)
            
            if orig_map50 and sugg_map50:
                improvement = sugg_map50 - orig_map50
                improvement_pct = (improvement / orig_map50 * 100) if orig_map50 > 0 else 0
                
                report["improvement"] = {
                    "absolute": improvement,
                    "percentage": improvement_pct
                }
        
        # 保存报告
        report_file = OUTPUT_ROOT / "dataset_variants_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 报告已保存: {report_file}")
        
        # 打印摘要
        print("\n" + "=" * 70)
        print("实验摘要")
        print("=" * 70)
        print(f"原始数据集 mAP@0.5: {results.get('original', {}).get('mAP50', 'N/A')}")
        print(f"Suggestions数据集 mAP@0.5: {results.get('suggestions', {}).get('mAP50', 'N/A')}")
        if "improvement" in report:
            print(f"改进: {report['improvement']['absolute']:+.4f} ({report['improvement']['percentage']:+.2f}%)")
        print("=" * 70)
        
        print("\n✅ 实验完成!")
        print(f"报告文件: {report_file}")
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

