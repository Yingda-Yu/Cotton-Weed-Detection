#!/usr/bin/env python3
"""
标准训练脚本 - 不依赖3LC
使用标准Ultralytics YOLO进行训练

Usage:
    python train_standard.py
    python train_standard.py --data dataset.yaml --epochs 30 --batch 16
"""

import torch
from ultralytics import YOLO
import argparse
from pathlib import Path
import gc

# ============================================================================
# CONFIGURATION - 默认配置（可通过命令行参数覆盖）
# ============================================================================

# 数据集配置
DATASET_YAML = "dataset.yaml"  # 默认使用原始数据集
# DATASET_YAML = "dataset_cleaned.yaml"  # 使用清洗后的数据集

# 训练超参数
EPOCHS = 30  # 训练轮数
BATCH_SIZE = 16  # 批次大小
IMAGE_SIZE = 640  # 输入图像尺寸（竞赛要求固定）
DEVICE = 0  # GPU设备（0表示第一块GPU，'cpu'表示CPU）
WORKERS = 4  # 数据加载器工作进程数

# 高级超参数
LR0 = 0.01  # 初始学习率
PATIENCE = 20  # 早停耐心值（无改进的轮数）

# 数据增强
USE_AUGMENTATION = False  # 是否启用增强（mosaic, mixup等）

# 模型配置
MODEL_WEIGHTS = "yolov8n.pt"  # 预训练权重
PROJECT_NAME = "runs/detect"  # 项目目录
RUN_NAME = "yolov8n_standard"  # 运行名称

# ============================================================================
# 训练函数
# ============================================================================

def main():
    """主训练流程"""
    parser = argparse.ArgumentParser(description="标准YOLOv8训练脚本")
    parser.add_argument("--data", type=str, default=DATASET_YAML, help="数据集YAML文件路径")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="训练轮数")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE, help="图像尺寸")
    parser.add_argument("--device", default=DEVICE, help="设备（GPU编号或'cpu'）")
    parser.add_argument("--workers", type=int, default=WORKERS, help="数据加载器工作进程数")
    parser.add_argument("--lr0", type=float, default=LR0, help="初始学习率")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="早停耐心值")
    parser.add_argument("--augment", action="store_true", help="启用数据增强")
    parser.add_argument("--model", type=str, default=MODEL_WEIGHTS, help="预训练模型权重")
    parser.add_argument("--name", type=str, default=RUN_NAME, help="运行名称")
    parser.add_argument("--project", type=str, default=PROJECT_NAME, help="项目目录")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的checkpoint路径")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("COTTON WEED DETECTION - 标准训练（不依赖3LC）")
    print("=" * 70)
    
    # 检查环境
    print("\n环境信息:")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
    
    # 检查数据集文件
    dataset_path = Path(args.data)
    if not dataset_path.exists():
        print(f"\n❌ 错误: 找不到数据集配置文件: {args.data}")
        print(f"   当前目录: {Path.cwd()}")
        print(f"   请确保数据集YAML文件存在")
        return
    
    print(f"\n✅ 数据集配置: {args.data}")
    
    # 显示训练配置
    print("\n" + "=" * 70)
    print("训练配置")
    print("=" * 70)
    print(f"  运行名称: {args.name}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch}")
    print(f"  图像尺寸: {args.imgsz}")
    print(f"  设备: {'GPU ' + str(args.device) if args.device != 'cpu' else 'CPU'}")
    print(f"  学习率: {args.lr0}")
    print(f"  早停耐心值: {args.patience}")
    print(f"  数据增强: {'启用' if args.augment else '禁用'}")
    
    # 加载模型
    print("\n" + "=" * 70)
    print("加载模型")
    print("=" * 70)
    # 加载模型（支持恢复训练）
    if args.resume:
        print(f"\n恢复训练: {args.resume}")
        model = YOLO(args.resume)
        print(f"✅ 从checkpoint恢复训练")
    else:
        print(f"\n加载预训练模型: {args.model}")
        model = YOLO(args.model)
        print(f"✅ 模型加载成功 (YOLOv8n, ~3M参数)")
    
    # 准备训练参数
    # 强制单进程模式以避免内存问题
    workers = 0 if args.workers == 0 else args.workers
    
    train_args = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": workers,  # 确保单进程模式
        "lr0": args.lr0,
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
        "val": True,  # 启用验证
        "save": True,  # 保存检查点
        "plots": True,  # 生成训练图表
        "verbose": True,  # 显示详细训练进度
    }
    
    # 如果workers=0，禁用多进程相关的增强以避免内存问题
    if workers == 0:
        print(f"\n⚠️  使用单进程模式 (workers=0) 以避免内存问题")
        # 禁用可能占用额外内存的增强
        train_args["mosaic"] = 0.0  # 禁用mosaic以减少内存占用
    
    # 如果提供了resume参数，添加resume标志
    if args.resume:
        train_args["resume"] = True
    
    # 添加数据增强（如果启用）
    if args.augment:
        train_args.update({
            "mosaic": 1.0,  # Mosaic增强
            "mixup": 0.05,  # Mixup增强
            "copy_paste": 0.1,  # Copy-paste增强
        })
        print("\n✅ 数据增强已启用")
    
    # 清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # 开始训练
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70 + "\n")
    
    try:
        results = model.train(**train_args)
        
        # 训练完成
        print("\n" + "=" * 70)
        print("✅ 训练完成！")
        print("=" * 70)
        print(f"\n模型权重保存位置:")
        print(f"  最佳模型: {args.project}/{args.name}/weights/best.pt")
        print(f"  最后模型: {args.project}/{args.name}/weights/last.pt")
        
        if hasattr(results, 'results_dict'):
            print(f"\n训练结果:")
            if 'metrics/mAP50' in results.results_dict:
                print(f"  mAP@0.5: {results.results_dict['metrics/mAP50']:.4f}")
            if 'metrics/mAP50-95' in results.results_dict:
                print(f"  mAP@0.5:0.95: {results.results_dict['metrics/mAP50-95']:.4f}")
        
        print(f"\n下一步:")
        print(f"  1. 查看训练结果: {args.project}/{args.name}/")
        print(f"  2. 生成预测: python predict.py --model {args.project}/{args.name}/weights/best.pt")
        print(f"  3. 提交到Kaggle: 上传 submission.csv")
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        
        # 如果是内存错误，提供解决建议
        error_str = str(e).lower()
        if "memory" in error_str or "insufficient" in error_str or "allocate" in error_str:
            print("\n" + "=" * 70)
            print("💡 内存不足解决方案")
            print("=" * 70)
            print("1. 减小batch size（当前: {}）: --batch 4 或 --batch 2".format(args.batch))
            print("2. 确保使用单进程: --workers 0（已设置）")
            print("3. 关闭其他占用内存的程序")
            print("4. 增加Windows页面文件大小:")
            print("   控制面板 > 系统 > 高级系统设置 > 性能设置 > 高级 > 虚拟内存")
            print("   建议设置为: 初始大小 8192MB, 最大大小 16384MB")
            print("5. 如果问题持续，尝试使用CPU训练: --device cpu")
            print("\n重新训练命令示例:")
            print(f"python train_standard.py --data {args.data} --epochs {args.epochs} --batch 4 --imgsz {args.imgsz} --device {args.device} --workers 0 --name {args.name} --resume {args.resume if args.resume else ''}")
        
        raise


if __name__ == "__main__":
    main()

