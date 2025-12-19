#!/usr/bin/env python3
"""
完整的数据清洗和训练流程
自动执行：baseline训练 -> 数据清洗 -> 清洗后训练 -> 性能对比

Usage:
    python run_complete_workflow.py
"""

import subprocess
import sys
import time
from pathlib import Path
import json

# 配置
EPOCHS = 30  # 训练轮数
BATCH_SIZE = 16  # 使用优化后的batch size
BASELINE_NAME = "yolov8n_baseline_fast2"  # 使用你刚训练的baseline模型
CLEANED_NAME = "yolov8n_cleaned_fast"  # 清洗后的模型名称
BASELINE_MODEL = f"runs/detect/{BASELINE_NAME}/weights/best.pt"

def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def wait_for_training_complete(model_path, max_wait=3600):
    """等待训练完成"""
    print(f"\n⏳ 等待训练完成: {model_path}")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        if Path(model_path).exists():
            print("✅ 训练完成！")
            return True
        time.sleep(10)
        print(".", end="", flush=True)
    
    print(f"\n⚠️  超时：训练可能仍在进行中")
    return False

def step1_train_baseline():
    """步骤1: 训练baseline模型"""
    print_section("步骤1: 训练Baseline模型")
    
    print(f"配置:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  数据集: dataset.yaml (原始训练集)")
    print(f"  输出: {BASELINE_NAME}")
    
    # 检查是否已存在
    if Path(BASELINE_MODEL).exists():
        print(f"\n✅ Baseline模型已存在: {BASELINE_MODEL}")
        print("   自动跳过baseline训练，使用已有模型")
        return True
    
    print(f"\n🚀 开始训练baseline模型...")
    cmd = [
        sys.executable,
        "train_standard.py",
        "--data", "dataset.yaml",
        "--epochs", str(EPOCHS),
        "--batch", str(BATCH_SIZE),
        "--imgsz", "640",
        "--device", "0",
        "--workers", "4",  # 使用优化后的workers
        "--name", BASELINE_NAME
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ Baseline训练失败")
        return False
    
    if Path(BASELINE_MODEL).exists():
        print(f"✅ Baseline训练完成: {BASELINE_MODEL}")
        return True
    else:
        print("❌ 训练完成但找不到模型文件")
        return False

def step2_analyze_quality():
    """步骤2: 分析训练集标签质量"""
    print_section("步骤2: 分析训练集标签质量")
    
    if not Path(BASELINE_MODEL).exists():
        print(f"❌ 找不到baseline模型: {BASELINE_MODEL}")
        return False
    
    print(f"使用baseline模型: {BASELINE_MODEL}")
    print(f"分析数据集: train (训练集)")
    
    cmd = [
        sys.executable,
        "tools/run_label_quality_analysis.py",
        "--model", BASELINE_MODEL,
        "--split", "train"
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ 标签质量分析失败")
        return False
    
    quality_report = "quality_report_train.json"
    if Path(quality_report).exists():
        print(f"✅ 质量分析完成: {quality_report}")
        return True
    else:
        print("❌ 分析完成但找不到报告文件")
        return False

def step3_clean_dataset():
    """步骤3: 清洗训练集"""
    print_section("步骤3: 清洗训练集标注")
    
    quality_report = "quality_report_train.json"
    predictions_file = "predictions_train_coco.json"
    
    if not Path(quality_report).exists():
        print(f"❌ 找不到质量报告: {quality_report}")
        print("   请先运行步骤2")
        return False
    
    if not Path(predictions_file).exists():
        print(f"❌ 找不到预测文件: {predictions_file}")
        print("   请先运行步骤2")
        return False
    
    print(f"使用质量报告: {quality_report}")
    print(f"使用预测文件: {predictions_file}")
    
    cmd = [
        sys.executable,
        "tools/clean_dataset.py",
        "--quality-report", quality_report,
        "--predictions", predictions_file,
        "--output", "cleaned_train_annotations.json"
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ 数据清洗失败")
        return False
    
    cleaned_file = "cleaned_train_annotations.json"
    if Path(cleaned_file).exists():
        print(f"✅ 数据清洗完成: {cleaned_file}")
        return True
    else:
        print("❌ 清洗完成但找不到输出文件")
        return False

def step4_convert_and_prepare():
    """步骤4: 转换格式并准备清洗后的数据集"""
    print_section("步骤4: 准备清洗后的数据集")
    
    # 使用run_cleaning_and_comparison.py的步骤2-4
    print("执行格式转换和文件准备...")
    
    # 这里可以调用run_cleaning_and_comparison.py的相关函数
    # 或者直接执行命令
    try:
        from tools.run_cleaning_and_comparison import (
            step2_convert_to_yolo,
            step3_copy_images,
            step4_create_dataset_yaml
        )
        
        # 步骤2: 转换为YOLO格式
        print("\n[4.1] 转换为YOLO格式...")
        labels_dir = step2_convert_to_yolo()
        if not labels_dir:
            return False
        
        # 步骤3: 复制图片
        print("\n[4.2] 复制图片文件...")
        if not step3_copy_images():
            return False
        
        # 步骤4: 创建数据集配置
        print("\n[4.3] 创建数据集配置...")
        yaml_file = step4_create_dataset_yaml()
        if not yaml_file:
            return False
        
        print(f"✅ 数据集准备完成: {yaml_file}")
        return True
        
    except Exception as e:
        print(f"❌ 准备数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def step5_train_cleaned():
    """步骤5: 使用清洗后的数据训练"""
    print_section("步骤5: 使用清洗后的数据训练")
    
    dataset_yaml = "dataset_cleaned.yaml"
    if not Path(dataset_yaml).exists():
        print(f"❌ 找不到数据集配置: {dataset_yaml}")
        print("   请先运行步骤4")
        return False
    
    print(f"配置:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  数据集: {dataset_yaml} (清洗后的训练集)")
    print(f"  输出: {CLEANED_NAME}")
    
    print(f"\n🚀 开始训练清洗后的模型...")
    cmd = [
        sys.executable,
        "train_standard.py",
        "--data", dataset_yaml,
        "--epochs", str(EPOCHS),
        "--batch", str(BATCH_SIZE),
        "--imgsz", "640",
        "--device", "0",
        "--workers", "4",  # 使用优化后的workers
        "--name", CLEANED_NAME
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ 清洗后训练失败")
        return False
    
    cleaned_model = f"runs/detect/{CLEANED_NAME}/weights/best.pt"
    if Path(cleaned_model).exists():
        print(f"✅ 清洗后训练完成: {cleaned_model}")
        return True
    else:
        print("❌ 训练完成但找不到模型文件")
        return False

def step6_compare_performance():
    """步骤6: 对比性能"""
    print_section("步骤6: 性能对比")
    
    baseline_model = BASELINE_MODEL
    cleaned_model = f"runs/detect/{CLEANED_NAME}/weights/best.pt"
    
    # 读取baseline结果
    baseline_results = Path(baseline_model).parent.parent / "results.csv"
    cleaned_results = Path(cleaned_model).parent.parent / "results.csv"
    
    baseline_map = None
    cleaned_map = None
    
    # 读取baseline mAP
    if baseline_results.exists():
        try:
            import pandas as pd
            df = pd.read_csv(baseline_results)
            if len(df) > 0:
                last_row = df.iloc[-1]
                for col in df.columns:
                    if 'map50' in col.lower() and 'metrics' in col.lower():
                        baseline_map = last_row.get(col, None)
                        break
        except:
            pass
    
    # 读取cleaned mAP
    if cleaned_results.exists():
        try:
            import pandas as pd
            df = pd.read_csv(cleaned_results)
            if len(df) > 0:
                last_row = df.iloc[-1]
                for col in df.columns:
                    if 'map50' in col.lower() and 'metrics' in col.lower():
                        cleaned_map = last_row.get(col, None)
                        break
        except:
            pass
    
    print(f"\n📊 性能对比:")
    print(f"  Baseline模型: {baseline_model}")
    if baseline_map:
        print(f"    mAP@0.5: {baseline_map:.4f} ({baseline_map*100:.2f}%)")
    else:
        print(f"    mAP@0.5: 无法读取")
    
    print(f"\n  清洗后模型: {cleaned_model}")
    if cleaned_map:
        print(f"    mAP@0.5: {cleaned_map:.4f} ({cleaned_map*100:.2f}%)")
    else:
        print(f"    mAP@0.5: 无法读取")
    
    if baseline_map and cleaned_map:
        improvement = cleaned_map - baseline_map
        improvement_pct = (improvement / baseline_map * 100) if baseline_map > 0 else 0
        print(f"\n  ✅ 性能提升:")
        print(f"    绝对提升: {improvement:+.4f} ({improvement*100:+.2f}%)")
        print(f"    相对提升: {improvement_pct:+.2f}%")
        
        # 保存对比报告
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "baseline": {
                "model": str(baseline_model),
                "mAP50": float(baseline_map)
            },
            "cleaned": {
                "model": str(cleaned_model),
                "mAP50": float(cleaned_map)
            },
            "improvement": {
                "absolute": float(improvement),
                "percentage": float(improvement_pct)
            }
        }
        
        report_file = "complete_workflow_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 对比报告已保存: {report_file}")
    
    return True

def main():
    """主流程"""
    print("=" * 70)
    print("完整的数据清洗和训练流程")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Baseline名称: {BASELINE_NAME}")
    print(f"  清洗后名称: {CLEANED_NAME}")
    
    steps = [
        ("训练Baseline", step1_train_baseline),
        ("分析标签质量", step2_analyze_quality),
        ("清洗数据集", step3_clean_dataset),
        ("准备清洗后的数据集", step4_convert_and_prepare),
        ("训练清洗后的模型", step5_train_cleaned),
        ("性能对比", step6_compare_performance),
    ]
    
    for i, (name, func) in enumerate(steps, 1):
        print(f"\n{'='*70}")
        print(f"执行步骤 {i}/{len(steps)}: {name}")
        print(f"{'='*70}")
        
        if not func():
            print(f"\n❌ 步骤 {i} 失败，流程终止")
            return False
        
        print(f"\n✅ 步骤 {i} 完成")
    
    print("\n" + "=" * 70)
    print("✅ 完整流程执行成功！")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

