#!/usr/bin/env python3
"""
实验1: Noise Impact（噪声影响实验）
训练不同噪声数据集上的模型，评估mAP@0.5，并绘制图表

用法:
    python experiments/experiment1_noise_impact.py
"""

import json
import os
import yaml
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import torch
import sys

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ================================
# 配置
# ================================
WORKSPACE_ROOT = Path(__file__).parent.parent  # 项目根目录
NOISY_DATASETS_ROOT = WORKSPACE_ROOT / "dataprocess" / "cottonweed_split" / "train" / "noisy datasets"
VAL_DIR = WORKSPACE_ROOT / "cotton weed dataset" / "val"  # 验证集目录（原始，无噪声）
OUTPUT_ROOT = WORKSPACE_ROOT / "experiments" / "experiment1_results"

MODEL_WEIGHTS = "yolov8n.pt"
EPOCHS = 5
BATCH_SIZE = 16
IMAGE_SIZE = 640
DEVICE = 0

# 噪声类型映射（用于图表显示）
NOISE_TYPE_MAPPING = {
    "missing": "missing",
    "spurious": "spurious", 
    "mislocated": "location",
    "mislabeled": "label-uniform"
}

# 类别映射
CLASS_NAME_TO_ID = {
    "carpetweed": 0,
    "morningglory": 1,
    "palmer_amaranth": 2
}

# ================================
# VIA 到 YOLO 转换函数
# ================================
def via_to_yolo(via_annot_dir, images_dir, output_labels_dir):
    """将VIA格式标注转换为YOLO格式"""
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    converted = 0
    skipped = 0
    
    for via_file in Path(via_annot_dir).glob("*.json"):
        try:
            # 读取VIA格式
            with open(via_file, "r", encoding="utf-8") as f:
                via_data = json.load(f)
            
            # 获取图像文件名
            via_key = list(via_data.keys())[0]
            filename = via_data[via_key]["filename"]
            base_name = Path(filename).stem
            
            # 查找对应的图像文件
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                potential = Path(images_dir) / (base_name + ext)
                if potential.exists():
                    img_path = potential
                    break
            
            if img_path is None:
                skipped += 1
                continue
            
            # 获取图像尺寸
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception as e:
                print(f"警告: 无法读取图像 {img_path}: {e}")
                skipped += 1
                continue
            
            # 转换标注
            regions = via_data[via_key].get("regions", [])
            yolo_lines = []
            
            for region in regions:
                shape_attrs = region.get("shape_attributes", {})
                region_attrs = region.get("region_attributes", {})
                
                if shape_attrs.get("name") != "rect":
                    continue
                
                x = shape_attrs.get("x", 0)
                y = shape_attrs.get("y", 0)
                w = shape_attrs.get("width", 0)
                h = shape_attrs.get("height", 0)
                
                class_name = region_attrs.get("class", "")
                class_id = CLASS_NAME_TO_ID.get(class_name, 0)
                
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
                
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            # 保存YOLO格式文件
            label_file = output_labels_dir / f"{base_name}.txt"
            with open(label_file, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))
            
            converted += 1
        except Exception as e:
            print(f"警告: 处理 {via_file} 时出错: {e}")
            skipped += 1
            continue
    
    return converted, skipped

# ================================
# 创建数据集配置文件
# ================================
def create_dataset_yaml(train_dir, output_yaml, val_dir=None):
    """为噪声数据集创建dataset.yaml文件"""
    if val_dir is None:
        val_dir = VAL_DIR / "images"
    
    config = {
        "path": str(WORKSPACE_ROOT.absolute()),
        "train": str(Path(train_dir).relative_to(WORKSPACE_ROOT) / "images"),
        "val": str(Path(val_dir).relative_to(WORKSPACE_ROOT) if Path(val_dir).is_absolute() else val_dir),
        "nc": 3,
        "names": {
            0: "carpetweed",
            1: "morningglory",
            2: "palmer_amaranth"
        }
    }
    
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return output_yaml

# ================================
# 转换所有噪声数据集
# ================================
def convert_all_noisy_datasets():
    """将所有VIA格式的噪声数据集转换为YOLO格式"""
    print("=" * 70)
    print("步骤1: 转换噪声数据集格式 (VIA -> YOLO)")
    print("=" * 70)
    
    converted_datasets = []
    
    for noise_dir in NOISY_DATASETS_ROOT.iterdir():
        if not noise_dir.is_dir():
            continue
        
        noise_name = noise_dir.name
        print(f"\n处理: {noise_name}")
        
        via_annot_dir = noise_dir / "annotations"
        images_dir = noise_dir / "images"
        labels_dir = noise_dir / "labels"
        
        if not via_annot_dir.exists() or not images_dir.exists():
            print(f"  跳过: 缺少必要目录")
            continue
        
        # 转换标注
        converted, skipped = via_to_yolo(via_annot_dir, images_dir, labels_dir)
        print(f"  转换: {converted} 个标注文件, 跳过: {skipped} 个")
        
        # 创建dataset.yaml
        dataset_yaml = noise_dir / "dataset.yaml"
        create_dataset_yaml(noise_dir, dataset_yaml)
        print(f"  创建配置文件: {dataset_yaml}")
        
        converted_datasets.append({
            "name": noise_name,
            "path": noise_dir,
            "yaml": dataset_yaml
        })
    
    print(f"\n✅ 完成! 共转换 {len(converted_datasets)} 个数据集")
    return converted_datasets

# ================================
# 训练模型
# ================================
def train_model(dataset_yaml, run_name, epochs=EPOCHS, batch=BATCH_SIZE):
    """训练YOLO模型"""
    print(f"\n训练模型: {run_name}")
    print(f"  数据集: {dataset_yaml}")
    print(f"  Epochs: {epochs}, Batch: {batch}")
    
    model = YOLO(MODEL_WEIGHTS)
    
    try:
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            batch=batch,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            workers=0,  # 避免内存问题
            project=str(OUTPUT_ROOT / "runs"),
            name=run_name,
            val=True,
            save=True,
            plots=True
        )
        
        # 提取mAP@0.5
        map50 = None
        if hasattr(results, 'results_dict'):
            map50 = results.results_dict.get('metrics/mAP50(B)', None)
        
        # 如果无法从results获取，尝试从results.csv读取
        if map50 is None:
            results_csv = OUTPUT_ROOT / "runs" / run_name / "results.csv"
            if results_csv.exists():
                try:
                    df = pd.read_csv(results_csv)
                    if 'metrics/mAP50(B)' in df.columns:
                        map50 = df['metrics/mAP50(B)'].iloc[-1]
                except:
                    pass
        
        print(f"  ✅ 训练完成, mAP@0.5: {map50:.4f}" if map50 else "  ✅ 训练完成")
        return map50
        
    except Exception as e:
        print(f"  ❌ 训练失败: {e}")
        return None

# ================================
# 批量训练所有噪声数据集
# ================================
def train_all_models(datasets, baseline_map=None):
    """训练所有噪声数据集上的模型"""
    print("\n" + "=" * 70)
    print("步骤2: 训练模型")
    print("=" * 70)
    
    results = {}
    
    # 添加基线结果（0%噪声）
    if baseline_map is not None:
        results["baseline"] = {
            "noise_type": "baseline",
            "noise_ratio": 0,
            "map50": baseline_map
        }
        print(f"\n基线模型 mAP@0.5: {baseline_map:.4f}")
    
    # 训练每个噪声数据集
    for dataset in datasets:
        noise_name = dataset["name"]
        dataset_yaml = dataset["yaml"]
        run_name = f"noise_{noise_name}"
        
        map50 = train_model(dataset_yaml, run_name)
        
        if map50 is not None:
            # 解析噪声类型和比例
            parts = noise_name.split("_")
            if len(parts) == 2:
                noise_type = parts[0]
                ratio_str = parts[1]
                ratio = int(ratio_str) / 100.0
                
                results[noise_name] = {
                    "noise_type": noise_type,
                    "noise_ratio": ratio,
                    "map50": map50
                }
    
    return results

# ================================
# 绘制图表
# ================================
def plot_noise_impact(results, output_file=None):
    """绘制噪声影响图表"""
    print("\n" + "=" * 70)
    print("步骤3: 绘制图表")
    print("=" * 70)
    
    if output_file is None:
        output_file = OUTPUT_ROOT / "noise_impact.png"
    
    # 组织数据
    noise_types = ["missing", "spurious", "mislocated", "mislabeled"]
    ratios = [0, 0.05, 0.10, 0.20]
    
    # 准备绘图数据
    plot_data = {}
    for noise_type in noise_types:
        plot_data[noise_type] = []
        for ratio in ratios:
            if ratio == 0:
                # 基线值
                map50 = results.get("baseline", {}).get("map50", None)
            else:
                # 查找对应的噪声数据集结果
                dataset_name = f"{noise_type}_{int(ratio*100)}"
                map50 = results.get(dataset_name, {}).get("map50", None)
            plot_data[noise_type].append(map50)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 定义颜色和线型
    colors = {
        "missing": "#1f77b4",
        "spurious": "#ff7f0e",
        "mislocated": "#2ca02c",
        "mislabeled": "#d62728"
    }
    
    markers = {
        "missing": "o",
        "spurious": "s",
        "mislocated": "^",
        "mislabeled": "v"
    }
    
    # 绘制每条线
    for noise_type in noise_types:
        values = plot_data[noise_type]
        display_name = NOISE_TYPE_MAPPING.get(noise_type, noise_type)
        
        # 过滤掉None值
        valid_ratios = []
        valid_values = []
        for i, (r, v) in enumerate(zip(ratios, values)):
            if v is not None:
                valid_ratios.append(r * 100)  # 转换为百分比
                valid_values.append(v)
        
        if len(valid_values) > 0:
            ax.plot(valid_ratios, valid_values, 
                   marker=markers[noise_type],
                   color=colors[noise_type],
                   label=display_name,
                   linewidth=2,
                   markersize=8)
    
    # 设置图表属性
    ax.set_xlabel("Noisy labels fraction [%]", fontsize=12)
    ax.set_ylabel("mAP@0.5", fontsize=12)
    ax.set_title("Influence of different types of noise in the training dataset on model quality", 
                 fontsize=13, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim(-1, 21)
    
    # 设置x轴刻度
    ax.set_xticks([0, 5, 10, 20])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图表已保存: {output_file}")
    
    return fig

# ================================
# 主函数
# ================================
def main():
    """主流程"""
    print("=" * 70)
    print("实验1: Noise Impact（噪声影响实验）")
    print("=" * 70)
    
    # 创建输出目录
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # 检查基线模型
    baseline_model = WORKSPACE_ROOT / "runs" / "detect" / "yolov8n_baseline_new" / "weights" / "best.pt"
    baseline_map = None
    
    if baseline_model.exists():
        print(f"\n找到基线模型: {baseline_model}")
        # 尝试从complete_workflow_report.json读取基线mAP
        report_file = WORKSPACE_ROOT / "complete_workflow_report.json"
        if report_file.exists():
            try:
                with open(report_file, "r") as f:
                    report = json.load(f)
                    baseline_map = report.get("baseline", {}).get("mAP50", None)
                    if baseline_map:
                        print(f"基线 mAP@0.5: {baseline_map:.4f}")
            except:
                pass
    
    # 步骤1: 转换数据集格式
    datasets = convert_all_noisy_datasets()
    
    if len(datasets) == 0:
        print("\n❌ 没有找到可转换的数据集")
        return
    
    # 询问是否继续训练
    print("\n" + "=" * 70)
    response = input(f"准备训练 {len(datasets)} 个模型，每个 {EPOCHS} epochs。\n是否继续? (y/n): ").strip().lower()
    if response != 'y':
        print("已取消")
        return
    
    # 步骤2: 训练模型
    results = train_all_models(datasets, baseline_map)
    
    # 保存结果
    results_file = OUTPUT_ROOT / "noise_impact_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 结果已保存: {results_file}")
    
    # 步骤3: 绘制图表
    if len(results) > 1:
        plot_noise_impact(results)
        print("\n" + "=" * 70)
        print("✅ 完成!")
        print("=" * 70)
    else:
        print("\n⚠️  结果数据不足，无法绘制图表")

if __name__ == "__main__":
    main()

