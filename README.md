# 🌾 Cotton Weed Detection - 棉花杂草检测项目

基于YOLOv8的数据中心AI方法，使用SafeDNN-Clean进行自动数据清洗和标签质量提升。

---

## 📋 目录

- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [核心功能](#核心功能)
- [数据清洗流程](#数据清洗流程)
- [数据集处理](#数据集处理)
- [详细使用说明](#详细使用说明)
- [常见问题](#常见问题)

---

## 🚀 快速开始

### 1. 训练模型

```bash
# 使用原始数据训练
python train_standard.py --data dataset.yaml --epochs 30

# 使用清洗后的数据训练
python train_standard.py --data dataset_cleaned.yaml --epochs 30
```

### 2. 生成预测

```bash
python predict.py --model runs/detect/xxx/weights/best.pt
```

### 3. 完整工作流程（推荐）

```bash
# 一键完成：baseline训练 → 数据清洗 → 清洗后训练 → 性能对比
python run_complete_workflow.py
```

---

## 📁 项目结构

```
Cotton Weed Detect/
├── train_standard.py          # 核心：训练脚本
├── predict.py                 # 核心：预测脚本
├── run_complete_workflow.py  # 核心：完整工作流程
│
├── tools/                     # 工具类脚本
│   ├── run_label_quality_analysis.py  # 标签质量分析
│   ├── clean_dataset.py               # 数据清洗
│   ├── run_cleaning_and_comparison.py # 清洗对比流程
│   ├── visualize_quality_report.py    # 可视化质量报告
│   └── visualize_annotations.py       # 可视化标注
│
├── dataset/                   # 数据集处理工具
│   ├── yolo_to_coco.py        # YOLO转COCO格式
│   ├── coco_to_yolo.py        # COCO转YOLO格式
│   └── generate_predictions_coco.py  # 生成COCO格式预测
│
├── dataset.yaml               # 数据集配置
├── dataset_cleaned.yaml       # 清洗后数据集配置
│
├── train/                     # 训练集
├── val/                       # 验证集
├── test/                      # 测试集
├── cleaned_train/             # 清洗后的训练集
│
├── runs/                      # 训练结果
├── yolov8n.pt                 # 预训练权重
│
└── otherwork/safednn-clean/   # SafeDNN-Clean框架
```

---

## 🎯 核心功能

### 训练模型

**基本使用：**
```bash
python train_standard.py --data dataset.yaml --epochs 30 --batch 8
```

**参数说明：**
- `--data`: 数据集配置文件（dataset.yaml 或 dataset_cleaned.yaml）
- `--epochs`: 训练轮数（默认30）
- `--batch`: 批次大小（默认16）
- `--imgsz`: 图像尺寸（默认640）
- `--device`: 设备（0表示GPU，'cpu'表示CPU）
- `--workers`: 数据加载进程数（默认4，内存不足时设为0）
- `--name`: 运行名称（默认yolov8n_standard）
- `--resume`: 从checkpoint恢复训练

**示例：**
```bash
# 从checkpoint恢复训练
python train_standard.py \
    --data dataset_cleaned.yaml \
    --epochs 30 \
    --batch 4 \
    --workers 0 \
    --resume runs/detect/yolov8n_cleaned_train/weights/last.pt
```

### 生成预测

**基本使用：**
```bash
python predict.py --model runs/detect/xxx/weights/best.pt
```

**参数说明：**
- `--model`: 模型权重路径
- `--conf`: 置信度阈值（默认0.25）
- `--output`: 输出CSV文件（默认submission.csv）

---

## 🔧 数据清洗流程

### 完整工作流程

使用 `run_complete_workflow.py` 一键完成所有步骤：

```bash
python run_complete_workflow.py
```

这会自动执行：
1. 训练baseline模型
2. 分析训练集标签质量
3. 清洗训练集标注
4. 准备清洗后的数据集
5. 使用清洗后的数据训练
6. 性能对比

### 分步执行

#### 步骤1: 标签质量分析

```bash
python tools/run_label_quality_analysis.py \
    --model runs/detect/yolov8n_baseline/weights/best.pt \
    --split train
```

**参数说明：**
- `--model`: 模型权重路径
- `--split`: 数据集分割（train或val）
- `--iou`: IoU聚类阈值（默认0.5）
- `--threshold`: 质量分数阈值（默认0.5）
- `--conf`: 预测置信度阈值（默认0.25）

**输出：**
- `quality_report_train.json` - 质量分析报告
- `predictions_train_coco.json` - 预测结果

#### 步骤2: 自动清洗数据

```bash
python tools/clean_dataset.py \
    --quality-report quality_report_train.json \
    --predictions predictions_train_coco.json \
    --output cleaned_train_annotations.json \
    --location-threshold 0.7 \
    --label-threshold 0.8 \
    --missing-threshold 0.5
```

**参数说明：**
- `--quality-report`: 质量报告文件
- `--predictions`: 预测结果文件
- `--output`: 输出文件
- `--location-threshold`: Location修复的预测分数阈值（默认0.7）
- `--label-threshold`: Label修复的预测分数阈值（默认0.8）
- `--missing-threshold`: Missing添加的预测分数阈值（默认0.5）

**清洗规则：**
1. **Spurious（虚假标注）**: 直接删除
2. **Location（定位错误）**: 用模型预测框替换（当预测分数≥阈值）
3. **Label（类别错误）**: 用模型预测类别替换（当预测分数≥阈值）
4. **Missing（缺失标注）**: 添加模型预测框（当预测分数≥阈值）

#### 步骤3: 转换格式并准备数据集

```bash
# 转换为YOLO格式
python dataset/coco_to_yolo.py \
    --coco-file cleaned_train_annotations.json \
    --split train \
    --output-dir cleaned_train

# 复制图片文件（Windows PowerShell）
Copy-Item -Path "train\images\*" -Destination "cleaned_train\images\" -Recurse
```

#### 步骤4: 使用清洗后的数据训练

```bash
python train_standard.py --data dataset_cleaned.yaml --epochs 30
```

### 可视化质量报告

```bash
python tools/visualize_quality_report.py \
    --report quality_report_train.json \
    --val-dir train \
    --output quality_issues \
    --top-n 50
```

---

## 📊 数据集处理

### 格式转换

#### YOLO转COCO

```bash
python dataset/yolo_to_coco.py --split train --output annotations_train_coco.json
```

#### COCO转YOLO

```bash
python dataset/coco_to_yolo.py \
    --coco-file cleaned_train_annotations.json \
    --split train \
    --output-dir cleaned_train
```

### 生成预测

```bash
python dataset/generate_predictions_coco.py \
    --model runs/detect/xxx/weights/best.pt \
    --split train \
    --annotations annotations_train_coco.json \
    --output predictions_train_coco.json \
    --conf 0.25
```

---

## 📖 详细使用说明

### 错误类型说明

SafeDNN-Clean会自动识别4种标注问题：

1. **Spurious（虚假标注）**
   - 含义：标注了但模型没检测到
   - 修复：删除误标注

2. **Missing（缺失标注）**
   - 含义：模型检测到了但没标注
   - 修复：添加缺失标注

3. **Location（定位错误）**
   - 含义：类别对但边界框位置不准
   - 修复：调整边界框位置

4. **Label（类别错误）**
   - 含义：检测到了但类别标注错误
   - 修复：修正类别标签

### 质量报告解读

质量报告 `quality_report_train.json` 包含：

```json
{
  "annotations": [
    {
      "id": 123,
      "image_id": 45,
      "category_id": 0,
      "bbox": [100, 200, 50, 60],
      "quality": 0.32,  // 质量分数（越低越差）
      "issue": "spurious"  // 问题类型
    }
  ]
}
```

- **质量分数（quality）**: 0-1，越低表示质量越差
- **问题类型（issue）**: spurious/missing/location/label

### 最佳实践

1. **优先级排序**：
   - 高置信度假阳性（conf > 0.7的missing）
   - 低质量分数标注（quality < 0.3）
   - 类别错误（label类型）
   - 定位错误（location类型）

2. **阈值调整策略**：
   - **保守策略**（高阈值）：只修复高置信度错误
   - **激进策略**（低阈值）：修复更多潜在错误

3. **迭代优化**：
   - 第一次清洗使用保守阈值
   - 根据性能提升调整阈值
   - 多次迭代直到性能稳定

---

## 🐛 常见问题

### Q1: SafeDNN-Clean脚本找不到？

**A**: 确保 `otherwork/safednn-clean/safednn-clean.py` 存在。

### Q2: cleanlab导入失败？

**A**: 安装cleanlab：
```bash
pip install cleanlab>=2.2.0
```

### Q3: 内存不足错误？

**A**: 
- 设置 `--workers 0`（单进程模式）
- 减小 `--batch` 大小
- 增加Windows页面文件大小

### Q4: 训练中断如何恢复？

**A**: 使用 `--resume` 参数：
```bash
python train_standard.py --resume runs/detect/xxx/weights/last.pt
```

### Q5: 清洗后性能下降？

**A**: 可能原因：
- 训练轮数不足（建议30 epochs）
- 标注删除过多
- 清洗阈值需要调整

---

## 📊 实验结果

本项目完成了6个核心实验，全面评估了CLOD（SafeDNN-Clean）方法在数据清洗和标签质量提升方面的有效性。

### 实验概览

| 实验 | 名称 | 主要发现 |
|------|------|---------|
| 实验1 | 噪声影响分析 | 评估不同噪声类型对模型性能的影响 |
| 实验2 | CLOD有效性评估 | CLOD在类别错误检测上AUROC=0.79，位置偏移检测AUROC=0.86 |
| 实验3 | CLOD vs SOTA | CLOD在25%噪声比例下，类别错误检测AUROC=0.88 |
| 实验4 | 数据集变体实验 | **应用CLOD建议后，模型mAP@0.5提升50.39%** |
| 实验5 | 人工检查实验 | 识别了493个问题标注（40.5%），其中264个为虚假标注 |
| 实验6 | IoU阈值分析 | 最佳IoU阈值为0.3，平均AUROC=0.67 |

### 实验2：CLOD有效性评估

**实验设置**：在验证集（255张图片，471个标注）上添加20%人工噪声

**关键结果**（IoU=0.5）：

| 噪声类型 | AUROC | 表现评价 |
|---------|-------|---------|
| **Label** (类别错误) | **0.7876** | ✅ 优秀 |
| **Location_20** (位置偏移20%) | **0.8571** | ✅ 优秀 |
| **Location_50** (位置偏移50%) | 0.0000 | ❌ 无法检测 |
| **Scale_20** (尺寸变化20%) | 0.5143 | ⚠️ 略好于随机 |
| **Scale_50** (尺寸变化50%) | 0.5875 | ⚠️ 中等 |
| **Spurious** (虚假标注) | 0.5000 | ❌ 随机水平 |
| **Missing** (缺失标注) | 0.0000 | ❌ 无法检测 |

**结论**：CLOD在检测类别错误和小幅位置偏移方面表现优秀，但在虚假标注和缺失标注检测上效果有限。

### 实验3：CLOD vs SOTA对比

**实验设置**：在验证集上添加25%人工噪声，对比CLOD与ObjectLab（SOTA方法）

**关键结果**：

| 噪声类型 | CLOD AUROC | 说明 |
|---------|-----------|------|
| **Label** | **0.8793** | 优秀表现 |
| **Location_20** | **0.8571** | 优秀表现 |
| **Location_50** | 0.0000 | 无法检测 |
| **Scale_20** | 0.5061 | 略好于随机 |
| **Scale_50** | 0.5931 | 中等表现 |
| **Spurious** | 0.5000 | 随机水平 |
| **Missing** | 0.0000 | 无法检测 |

**结论**：CLOD在类别错误检测上表现优秀（AUROC=0.88），在更高噪声比例下表现更好。

### 实验4：数据集变体实验 ⭐

**实验设置**：应用CLOD前20%的建议，训练模型并对比性能

**关键结果**：

| 数据集 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|--------|---------|--------------|-----------|--------|
| **原始数据集** | 0.3710 | 0.1894 | 0.6260 | 0.5839 |
| **Suggestions数据集** | **0.5579** | **0.4105** | **0.6708** | 0.5229 |
| **改进** | **+0.1869** | **+0.2211** | **+0.0448** | -0.0610 |
| **改进百分比** | **+50.39%** | **+116.79%** | **+7.15%** | -10.45% |

**数据集变化**：
- 原始标注数：1,061个
- Suggestions标注数：724个（删除了264个虚假标注）
- 应用的建议数：98个（前20%）

**结论**：通过CLOD自动清洗，模型性能提升超过50%，证明了数据质量对模型性能的重要影响。

### 实验5：人工检查实验

**实验设置**：在训练集（593张图片，1,061个标注）上运行CLOD分析

**关键结果**：

| 指标 | 数值 |
|------|------|
| **总标注数** | 1,217 |
| **有质量分数的标注** | 493 (40.5%) |
| **需要人工检查的标注** | 0 (0.00%) |

**问题类型分布**：

| 问题类型 | 数量 | 占比 |
|---------|------|------|
| **Spurious** (虚假标注) | 264 | 53.5% |
| **Missing** (缺失标注) | 156 | 31.6% |
| **Location** (定位错误) | 47 | 9.5% |
| **Label** (类别错误) | 26 | 5.3% |

**质量分数统计**：
- 最低：0.0400
- 最高：0.4992
- 平均：0.1119
- 中位数：0.0400
- 75%分位数：0.1331

**结论**：CLOD能够自动识别40.5%的标注存在问题，其中虚假标注是最主要的问题类型。

### 实验6：IoU阈值分析

**实验设置**：测试不同IoU阈值（0.3-0.7）对CLOD性能的影响

**关键结果**：

| IoU阈值 | 平均AUROC | 标准差 | 有效噪声类型数 |
|---------|----------|--------|---------------|
| **0.3** | **0.6695** | ±0.1654 | 4 |
| **0.4** | 0.6652 | ±0.1606 | 4 |
| **0.5** | 0.6007 | ±0.1323 | 3 |
| **0.6** | 0.6086 | ±0.1421 | 3 |
| **0.7** | 0.6234 | ±0.1380 | 3 |

**不同噪声类型的最佳IoU阈值**：

| 噪声类型 | 最佳IoU | 最佳AUROC |
|---------|---------|----------|
| **Label** | 0.7 | 0.8161 |
| **Location_25** | 0.3 | 0.8667 |
| **Scale_25** | 0.7 | 0.5541 |
| **Spurious** | 任意 | 0.5000 |
| **Missing** | 任意 | 0.0000 |

**结论**：
- 综合最佳IoU阈值：0.3（能检测更多噪声类型）
- Location噪声需要低IoU阈值（0.3-0.4）
- Label和Scale噪声需要高IoU阈值（0.6-0.7）
- 建议根据主要噪声类型选择合适的IoU阈值

### 实验总结

1. **CLOD有效性**：在类别错误和小幅位置偏移检测上表现优秀（AUROC>0.78）
2. **数据清洗价值**：应用CLOD建议后，模型性能提升50.39%
3. **问题分布**：虚假标注是最主要的问题类型（53.5%）
4. **IoU阈值选择**：根据噪声类型选择合适的IoU阈值，综合应用建议使用0.3

**详细实验结果**：所有实验结果保存在 `experiments/` 目录下，包括JSON报告和可视化图表。

---

## 📚 数据集信息

- **任务**: 多类别杂草检测（3个类别）
- **格式**: YOLO格式（归一化坐标）
- **模型**: YOLOv8n（固定，符合竞赛要求）
- **训练集**: 593张图片，1,061个标注
- **验证集**: 255张图片，471个标注
- **测试集**: 170张图片（无标注）

### 类别

- **0**: Carpetweed（地毯草）
- **1**: Morning Glory（牵牛花）
- **2**: Palmer Amaranth（长芒苋）

---

## 📝 提交格式

CSV文件，列名：`image_id,prediction_string`

**预测字符串格式**：
```
class_id confidence x_center y_center width height
```

**示例**：
```csv
image_id,prediction_string
20190613_6062W_CM_29,0 0.95 0.5 0.3 0.2 0.4 1 0.87 0.7 0.6 0.15 0.25
20200624_iPhone6_SY_132,no box
```

**要求**：
- 列名必须小写
- 坐标归一化到[0, 1]
- 无检测时使用 `no box`

---

## 🔗 参考资料

- [SafeDNN-Clean论文](https://arxiv.org/abs/2211.13993)
- [CleanLab文档](https://docs.cleanlab.ai/)
- [YOLOv8文档](https://docs.ultralytics.com/)
- [COCO格式说明](https://cocodataset.org/#format-data)

---

## 📄 许可证

数据集仅供竞赛使用，详见官方竞赛规则。

---

**开始使用：** `python run_complete_workflow.py` 🚀

