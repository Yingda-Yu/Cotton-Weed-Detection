#!/usr/bin/env python3
"""
完整的数据清洗和性能对比流程

自动完成：
1. 清洗数据集标注
2. 转换回YOLO格式
3. 复制图片文件
4. 使用清洗后的数据训练模型
5. 对比清洗前后的性能

⚠️ 注意：所有操作都不会修改原始数据集
"""

import json
import yaml
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse

# ============================================================================
# 配置参数
# ============================================================================

# 清洗参数（训练集）
QUALITY_REPORT = "quality_report_train.json"
PREDICTIONS_FILE = "predictions_train_coco.json"
CLEANED_ANNOTATIONS = "cleaned_train_annotations.json"

# 清洗阈值
LOCATION_THRESHOLD = 0.7
LABEL_THRESHOLD = 0.8
MISSING_THRESHOLD = 0.5

# 数据集路径（清洗训练集，验证集保持原始）
ORIGINAL_TRAIN_DIR = "train"
CLEANED_TRAIN_DIR = "cleaned_train"

# 训练配置
BASELINE_MODEL = "runs/detect/yolov8n_baseline_new/weights/best.pt"  # 用于生成预测的基线模型
EPOCHS = 10  # 训练轮数
BATCH_SIZE = 8  # 减小batch size避免内存问题
RUN_NAME_CLEANED = "yolov8n_cleaned_new"  # 清洗后训练的run名称

# 输出文件
COMPARISON_REPORT = "cleaning_comparison_report.json"


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def step1_clean_dataset():
    """步骤1: 清洗数据集"""
    print_section("步骤1: 自动清洗数据集标注")
    
    # 检查必要文件
    if not Path(QUALITY_REPORT).exists():
        print(f"❌ 错误: 找不到质量报告文件: {QUALITY_REPORT}")
        print("   请先运行: python run_label_quality_analysis.py --model <model_path> --split train")
        return False
    
    if not Path(PREDICTIONS_FILE).exists():
        print(f"❌ 错误: 找不到预测结果文件: {PREDICTIONS_FILE}")
        print("   请先运行: python run_label_quality_analysis.py --model <model_path> --split train")
        return False
    
    # 运行清洗脚本
    print(f"运行清洗脚本...")
    print(f"  质量报告: {QUALITY_REPORT}")
    print(f"  预测结果: {PREDICTIONS_FILE}")
    print(f"  输出文件: {CLEANED_ANNOTATIONS}")
    print(f"\n  清洗阈值:")
    print(f"    Location: {LOCATION_THRESHOLD}")
    print(f"    Label: {LABEL_THRESHOLD}")
    print(f"    Missing: {MISSING_THRESHOLD}")
    
    try:
        from tools.clean_dataset import clean_dataset
        
        cleaned_data = clean_dataset(
            quality_report_file=QUALITY_REPORT,
            predictions_file=PREDICTIONS_FILE,
            output_file=CLEANED_ANNOTATIONS,
            location_score_threshold=LOCATION_THRESHOLD,
            label_score_threshold=LABEL_THRESHOLD,
            missing_score_threshold=MISSING_THRESHOLD
        )
        
        # 统计清洗结果
        original_count = len([ann for ann in json.load(open(QUALITY_REPORT))["annotations"] 
                             if ann.get("id", 0) >= 0])
        cleaned_count = len(cleaned_data["annotations"])
        
        print(f"\n✅ 清洗完成!")
        print(f"   原始标注数: {original_count}")
        print(f"   清洗后标注数: {cleaned_count}")
        print(f"   净变化: {cleaned_count - original_count}")
        
        return {
            "original_annotations": original_count,
            "cleaned_annotations": cleaned_count,
            "net_change": cleaned_count - original_count
        }
        
    except Exception as e:
        print(f"❌ 清洗失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def step2_convert_to_yolo():
    """步骤2: 转换回YOLO格式"""
    print_section("步骤2: 转换清洗后的标注为YOLO格式")
    
    if not Path(CLEANED_ANNOTATIONS).exists():
        print(f"❌ 错误: 找不到清洗后的标注文件: {CLEANED_ANNOTATIONS}")
        return False
    
    try:
        from dataset.coco_to_yolo import coco_to_yolo
        
        output_labels_dir = coco_to_yolo(
            coco_file=CLEANED_ANNOTATIONS,
            split_dir=ORIGINAL_TRAIN_DIR,
            output_dir=CLEANED_TRAIN_DIR
        )
        
        # 统计转换结果
        label_files = list(output_labels_dir.glob("*.txt"))
        print(f"\n✅ 转换完成!")
        print(f"   生成的标注文件数: {len(label_files)}")
        print(f"   输出目录: {output_labels_dir.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def step3_copy_images():
    """步骤3: 复制图片文件"""
    print_section("步骤3: 复制图片文件到清洗后的数据集")
    
    original_images_dir = Path(ORIGINAL_TRAIN_DIR) / "images"
    cleaned_images_dir = Path(CLEANED_TRAIN_DIR) / "images"
    
    if not original_images_dir.exists():
        print(f"❌ 错误: 找不到原始图片目录: {original_images_dir}")
        return False
    
    # 创建输出目录
    cleaned_images_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制图片
    image_files = list(original_images_dir.glob("*.jpg"))
    print(f"找到 {len(image_files)} 张图片")
    
    copied = 0
    for img_file in image_files:
        dest = cleaned_images_dir / img_file.name
        if not dest.exists():
            shutil.copy2(img_file, dest)
            copied += 1
    
    print(f"\n✅ 复制完成!")
    print(f"   复制图片数: {copied}")
    print(f"   输出目录: {cleaned_images_dir.absolute()}")
    
    return True


def step4_create_dataset_yaml():
    """步骤4: 创建清洗后的数据集配置文件"""
    print_section("步骤4: 创建清洗后的数据集配置")
    
    # 读取原始配置
    with open("dataset.yaml", 'r', encoding='utf-8') as f:
        original_config = yaml.safe_load(f)
    
    # 创建新配置（使用清洗后的训练集，验证集保持原始）
    cleaned_config = original_config.copy()
    cleaned_config["train"] = f"{CLEANED_TRAIN_DIR}/images"
    cleaned_config["val"] = "val/images"  # 保持原始验证集，用于真实评估
    
    # 保存新配置
    cleaned_yaml = "dataset_cleaned.yaml"
    with open(cleaned_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(cleaned_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 配置文件已创建: {cleaned_yaml}")
    print(f"   训练集: {cleaned_config['train']} (清洗后)")
    print(f"   验证集: {cleaned_config['val']} (原始，用于真实评估)")
    
    return cleaned_yaml


def step5_train_with_cleaned_data(dataset_yaml):
    """步骤5: 使用清洗后的数据训练模型"""
    print_section("步骤5: 使用清洗后的数据训练模型")
    
    print(f"训练配置:")
    print(f"  数据集配置: {dataset_yaml}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  Run名称: {RUN_NAME_CLEANED}")
    
    # 检查是否已有训练结果
    weights_path = Path(f"runs/detect/{RUN_NAME_CLEANED}/weights/best.pt")
    if weights_path.exists():
        print(f"\n⚠️  警告: 发现已存在的训练结果: {weights_path}")
        response = input("   是否跳过训练，使用已有模型? (y/n): ").strip().lower()
        if response == 'y':
            print("   使用已有模型...")
            return str(weights_path)
    
    # 使用Ultralytics训练（不使用3LC，因为清洗后的数据不在3LC中）
    try:
        from ultralytics import YOLO
        
        print(f"\n开始训练...")
        model = YOLO("yolov8n.pt")
        
        results = model.train(
            data=dataset_yaml,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=640,
            name=RUN_NAME_CLEANED,
            project="runs/detect",
            device=0,
            workers=4
        )
        
        if weights_path.exists():
            print(f"\n✅ 训练完成!")
            print(f"   模型权重: {weights_path.absolute()}")
            
            # 提取最佳mAP（尝试多种方式）
            best_map = None
            try:
                # 方式1: 从results对象获取
                if hasattr(results, 'results_dict'):
                    best_map = results.results_dict.get('metrics/mAP50(B)', None)
                # 方式2: 从results.csv读取
                if best_map is None:
                    results_csv = weights_path.parent.parent / "results.csv"
                    if results_csv.exists():
                        try:
                            import pandas as pd
                            df = pd.read_csv(results_csv)
                            if len(df) > 0:
                                last_row = df.iloc[-1]
                                for col in df.columns:
                                    if 'map50' in col.lower() or 'mAP50' in col:
                                        best_map = last_row.get(col, None)
                                        break
                        except:
                            # 手动解析CSV
                            with open(results_csv, 'r') as f:
                                lines = f.readlines()
                                if len(lines) > 1:
                                    headers = lines[0].strip().split(',')
                                    last_line = lines[-1].strip().split(',')
                                    for i, h in enumerate(headers):
                                        if 'map50' in h.lower() or 'mAP50' in h:
                                            try:
                                                best_map = float(last_line[i])
                                            except:
                                                pass
                                            break
            except Exception as e:
                print(f"   警告: 无法自动提取mAP: {e}")
            
            if best_map is not None:
                print(f"   最佳mAP@0.5: {best_map:.4f}")
            else:
                print(f"   ⚠️  无法自动提取mAP，请手动查看训练日志")
                user_input = input("   请输入最佳mAP@0.5值（直接回车跳过）: ").strip()
                if user_input:
                    try:
                        best_map = float(user_input)
                    except ValueError:
                        best_map = None
            
            return {
                "weights_path": str(weights_path),
                "best_map": best_map,
                "epochs": EPOCHS
            }
        else:
            print(f"❌ 错误: 训练完成但找不到权重文件: {weights_path}")
            return None
            
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def step6_get_baseline_performance():
    """步骤6: 获取基线模型性能"""
    print_section("步骤6: 获取基线模型性能")
    
    baseline_weights = Path(BASELINE_MODEL)
    if not baseline_weights.exists():
        print(f"⚠️  警告: 找不到基线模型: {BASELINE_MODEL}")
        print("   将使用训练日志中的信息（如果有）")
        return None
    
    # 尝试从训练结果目录读取metrics
    baseline_run_dir = baseline_weights.parent.parent
    results_file = baseline_run_dir / "results.csv"
    
    best_map = None
    
    if results_file.exists():
        try:
            # 尝试使用pandas读取
            try:
                import pandas as pd
                df = pd.read_csv(results_file)
                if len(df) > 0:
                    # 获取最后一行的mAP
                    last_row = df.iloc[-1]
                    best_map = last_row.get('metrics/mAP50(B)', None)
                    if best_map is None or pd.isna(best_map):
                        # 尝试其他可能的列名
                        for col in df.columns:
                            if 'map50' in col.lower() or 'mAP50' in col:
                                best_map = last_row.get(col, None)
                                break
            except ImportError:
                # 如果没有pandas，手动解析CSV
                with open(results_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        # 第一行是表头
                        headers = lines[0].strip().split(',')
                        # 最后一行是数据
                        last_line = lines[-1].strip().split(',')
                        
                        # 查找mAP列
                        map_col_idx = None
                        for i, h in enumerate(headers):
                            if 'map50' in h.lower() or 'mAP50' in h:
                                map_col_idx = i
                                break
                        
                        if map_col_idx is not None and map_col_idx < len(last_line):
                            try:
                                best_map = float(last_line[map_col_idx])
                            except ValueError:
                                pass
        except Exception as e:
            print(f"⚠️  无法读取结果文件: {e}")
    
    if best_map is not None:
        print(f"✅ 基线性能:")
        print(f"   模型: {BASELINE_MODEL}")
        print(f"   最佳mAP@0.5: {best_map:.4f}")
        
        return {
            "weights_path": str(baseline_weights),
            "best_map": float(best_map)
        }
    else:
        print("⚠️  无法自动获取基线性能")
        print("   请手动输入基线模型的mAP@0.5值，或按Enter跳过")
        user_input = input("   基线mAP@0.5 (直接回车跳过): ").strip()
        if user_input:
            try:
                best_map = float(user_input)
                return {
                    "weights_path": str(baseline_weights),
                    "best_map": best_map,
                    "source": "manual_input"
                }
            except ValueError:
                print("   输入无效，跳过")
        
        return None


def step7_compare_performance(baseline_perf, cleaned_perf, cleaning_stats):
    """步骤7: 对比性能"""
    print_section("步骤7: 性能对比分析")
    
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "cleaning_stats": cleaning_stats,
        "baseline": baseline_perf,
        "cleaned": cleaned_perf
    }
    
    if baseline_perf and cleaned_perf:
        baseline_map = baseline_perf.get("best_map", 0)
        cleaned_map = cleaned_perf.get("best_map", 0)
        
        improvement = cleaned_map - baseline_map
        improvement_pct = (improvement / baseline_map * 100) if baseline_map > 0 else 0
        
        comparison["improvement"] = {
            "absolute": improvement,
            "percentage": improvement_pct
        }
        
        print(f"\n📊 性能对比:")
        print(f"   基线模型 mAP@0.5: {baseline_map:.4f}")
        print(f"   清洗后模型 mAP@0.5: {cleaned_map:.4f}")
        print(f"   绝对提升: {improvement:+.4f}")
        print(f"   相对提升: {improvement_pct:+.2f}%")
        
        if improvement > 0:
            print(f"\n   ✅ 清洗后性能提升!")
        elif improvement < 0:
            print(f"\n   ⚠️  清洗后性能下降，可能需要调整阈值")
        else:
            print(f"\n   ➡️  性能无明显变化")
    
    # 保存对比报告
    with open(COMPARISON_REPORT, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 对比报告已保存: {COMPARISON_REPORT}")
    
    return comparison


def main():
    """主流程"""
    global BASELINE_MODEL, EPOCHS, LOCATION_THRESHOLD, LABEL_THRESHOLD, MISSING_THRESHOLD
    
    parser = argparse.ArgumentParser(
        description="完整的数据清洗和性能对比流程"
    )
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="跳过清洗步骤（如果已经清洗过）"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="跳过训练步骤（如果已经训练过）"
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default=BASELINE_MODEL,
        help="基线模型路径（用于对比）"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="训练轮数"
    )
    parser.add_argument(
        "--location-threshold",
        type=float,
        default=LOCATION_THRESHOLD,
        help="Location修复阈值"
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=LABEL_THRESHOLD,
        help="Label修复阈值"
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=MISSING_THRESHOLD,
        help="Missing添加阈值"
    )
    
    args = parser.parse_args()
    
    # 更新全局配置
    BASELINE_MODEL = args.baseline_model
    EPOCHS = args.epochs
    LOCATION_THRESHOLD = args.location_threshold
    LABEL_THRESHOLD = args.label_threshold
    MISSING_THRESHOLD = args.missing_threshold
    
    print("=" * 70)
    print("  完整数据清洗和性能对比流程")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  基线模型: {BASELINE_MODEL}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  清洗阈值: Location={LOCATION_THRESHOLD}, Label={LABEL_THRESHOLD}, Missing={MISSING_THRESHOLD}")
    print(f"  输出目录: {CLEANED_TRAIN_DIR}")
    
    results = {}
    
    # 步骤1: 清洗数据集
    if not args.skip_cleaning:
        cleaning_stats = step1_clean_dataset()
        if cleaning_stats is None:
            print("\n❌ 清洗失败，流程终止")
            return
        results["cleaning_stats"] = cleaning_stats
    else:
        print("\n⏭️  跳过清洗步骤")
        # 尝试读取已有的清洗统计
        if Path(CLEANED_ANNOTATIONS).exists():
            with open(CLEANED_ANNOTATIONS, 'r', encoding='utf-8') as f:
                cleaned_data = json.load(f)
            results["cleaning_stats"] = {
                "cleaned_annotations": len(cleaned_data["annotations"])
            }
    
    # 步骤2: 转换回YOLO格式
    if not Path(CLEANED_TRAIN_DIR).exists() or not list(Path(CLEANED_TRAIN_DIR).glob("labels/*.txt")):
        if not step2_convert_to_yolo():
            print("\n❌ 转换失败，流程终止")
            return
    else:
        print("\n⏭️  跳过转换步骤（已存在清洗后的标注）")
    
    # 步骤3: 复制图片
    cleaned_images_dir = Path(CLEANED_TRAIN_DIR) / "images"
    if not cleaned_images_dir.exists() or not list(cleaned_images_dir.glob("*.jpg")):
        if not step3_copy_images():
            print("\n❌ 复制图片失败，流程终止")
            return
    else:
        print("\n⏭️  跳过复制图片步骤（已存在）")
    
    # 步骤4: 创建数据集配置
    dataset_yaml = step4_create_dataset_yaml()
    
    # 步骤5: 训练模型
    if not args.skip_training:
        cleaned_perf = step5_train_with_cleaned_data(dataset_yaml)
        if cleaned_perf is None:
            print("\n❌ 训练失败，流程终止")
            return
        results["cleaned_performance"] = cleaned_perf
    else:
        print("\n⏭️  跳过训练步骤")
        # 尝试读取已有模型
        weights_path = Path(f"runs/detect/{RUN_NAME_CLEANED}/weights/best.pt")
        if weights_path.exists():
            results["cleaned_performance"] = {
                "weights_path": str(weights_path),
                "note": "使用已有模型"
            }
    
    # 步骤6: 获取基线性能
    baseline_perf = step6_get_baseline_performance()
    if baseline_perf:
        results["baseline_performance"] = baseline_perf
    
    # 步骤7: 对比性能
    comparison = step7_compare_performance(
        baseline_perf,
        results.get("cleaned_performance"),
        results.get("cleaning_stats")
    )
    
    # 最终总结
    print("\n" + "=" * 70)
    print("  流程完成!")
    print("=" * 70)
    print(f"\n生成的文件:")
    print(f"  - 清洗后的标注: {CLEANED_ANNOTATIONS}")
    print(f"  - 清洗后的训练集: {CLEANED_TRAIN_DIR}/")
    print(f"  - 数据集配置: dataset_cleaned.yaml")
    print(f"  - 对比报告: {COMPARISON_REPORT}")
    print(f"\n重要说明:")
    print(f"  ✅ 训练集已清洗: {CLEANED_TRAIN_DIR}/")
    print(f"  ✅ 验证集保持原始: val/ (用于真实评估)")
    print(f"\n下一步:")
    print(f"  1. 查看对比报告: {COMPARISON_REPORT}")
    print(f"  2. 如果性能提升，可以继续优化清洗阈值")
    print(f"  3. 如果性能下降，可以调整阈值或手动检查清洗结果")
    print("=" * 70)


if __name__ == "__main__":
    main()

