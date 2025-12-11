# 实验脚本说明

本目录包含6个实验脚本，用于复现论文中的各项实验。

## 📋 实验列表

| 实验 | 脚本 | 状态 | 说明 |
|------|------|------|------|
| 实验1 | `experiment1_noise_impact.py` | ✅ 已存在 | 噪声影响分析 |
| 实验2 | `experiment2_clod_effectiveness.py` | ✅ 新建 | CLOD有效性评估 |
| 实验3 | `experiment3_clod_vs_sota.py` | ✅ 新建 | CLOD vs SOTA对比（CLOD部分） |
| 实验4 | `experiment4_dataset_variants.py` | ✅ 新建 | 数据集变体实验 |
| 实验5 | `experiment5_manual_inspection.py` | ✅ 新建 | 人工检查实验 |
| 实验6 | `experiment6_iou_threshold.py` | ✅ 新建 | IoU阈值分析 |

## 🚀 快速开始

### 前置要求

1. **已训练的baseline模型**
   ```bash
   # 如果还没有baseline模型，先训练一个
   python train_standard.py --data dataset.yaml --epochs 30 --name yolov8n_baseline_new
   ```

2. **安装依赖**
   ```bash
   pip install scikit-learn matplotlib pandas psutil
   ```

3. **确保SafeDNN-Clean可用**
   - 检查 `otherwork/safednn-clean/safednn-clean.py` 是否存在

### 运行实验

#### 实验5: Manual Inspection（推荐先运行，最简单）

```bash
python experiments/experiment5_manual_inspection.py \
    --model runs/detect/yolov8n_baseline_new/weights/best.pt \
    --split train
```

**输出:**
- `experiments/experiment5_results/manual_inspection_report.json` - 完整报告
- `experiments/experiment5_results/quality_distribution.png` - 质量分布图

---

#### 实验2: CLOD Effectiveness

```bash
python experiments/experiment2_clod_effectiveness.py \
    --model runs/detect/yolov8n_baseline_new/weights/best.pt \
    --split val \
    --noise-ratio 0.2
```

**输出:**
- `experiments/experiment2_results/clod_effectiveness_results.json` - 结果数据
- `experiments/experiment2_results/auroc_results.png` - AUROC对比图
- `experiments/experiment2_results/roc_curves.png` - ROC曲线
- `experiments/experiment2_results/iou_threshold_analysis.png` - IoU阈值分析

**说明:**
- 会在验证集上添加20%的人工噪声
- 测试5种噪声类型：label, location, scale, spurious, missing
- 计算AUROC评估CLOD的检测效果

---

#### 实验4: Dataset Variants

```bash
python experiments/experiment4_dataset_variants.py \
    --model runs/detect/yolov8n_baseline_new/weights/best.pt \
    --split train \
    --suggestions-ratio 0.2
```

**输出:**
- `experiments/experiment4_results/dataset_variants_report.json` - 完整报告
- `experiments/experiment4_results/runs/` - 训练结果

**说明:**
- 创建Suggestions数据集（应用CLOD前20%建议）
- 训练并对比原始数据集和Suggestions数据集的性能

---

#### 实验6: IoU Threshold Analysis

```bash
python experiments/experiment6_iou_threshold.py \
    --model runs/detect/yolov8n_baseline_new/weights/best.pt \
    --split val \
    --noise-ratio 0.2
```

**输出:**
- `experiments/experiment6_results/iou_threshold_results.json` - 结果数据
- `experiments/experiment6_results/iou_threshold_analysis.png` - 分析图
- `experiments/experiment6_results/iou_threshold_report.json` - 完整报告

**说明:**
- 测试不同IoU阈值（0.3-0.7）对CLOD性能的影响
- 找到最佳IoU阈值

---

#### 实验3: CLOD vs SOTA

```bash
python experiments/experiment3_clod_vs_sota.py \
    --model runs/detect/yolov8n_baseline_new/weights/best.pt \
    --split val \
    --noise-ratio 0.25
```

**输出:**
- `experiments/experiment3_results/clod_vs_sota_results.json` - 结果数据
- `experiments/experiment3_results/clod_vs_sota_comparison.png` - 对比图
- `experiments/experiment3_results/comparison_table.md` - 对比表格

**说明:**
- ⚠️ 注意：ObjectLab需要单独实现，当前只运行CLOD部分
- 对比CLOD和ObjectLab在检测人工噪声上的性能

---

#### 实验1: Noise Impact

```bash
python experiments/experiment1_noise_impact.py
```

**输出:**
- `experiments/experiment1_results/noise_impact_results.json` - 结果数据
- `experiments/experiment1_results/noise_impact.png` - 噪声影响图表
- `experiments/experiment1_results/runs/` - 训练结果

**说明:**
- 训练不同噪声数据集上的模型，评估mAP@0.5
- 绘制噪声类型与模型质量的关系图

---

## 📊 实验输出结构

```
experiments/
├── experiment2_results/
│   ├── clod_effectiveness_results.json
│   ├── auroc_results.png
│   ├── roc_curves.png
│   └── iou_threshold_analysis.png
├── experiment3_results/
│   ├── clod_vs_sota_results.json
│   ├── clod_vs_sota_comparison.png
│   └── comparison_table.md
├── experiment4_results/
│   ├── dataset_variants_report.json
│   └── runs/
│       ├── original/
│       └── suggestions/
├── experiment5_results/
│   ├── manual_inspection_report.json
│   └── quality_distribution.png
└── experiment6_results/
    ├── iou_threshold_results.json
    ├── iou_threshold_analysis.png
    └── iou_threshold_report.json
```

## 🔧 辅助模块

### 人工噪声生成模块

`dataprocess/add_artificial_noise.py` - 用于在COCO格式数据集上添加人工噪声

**支持的噪声类型:**
- `label`: 类别错误（随机替换类别）
- `location`: 位置偏移（25%或50%的框尺寸）
- `scale`: 尺寸变化（25%或50%的框尺寸）
- `spurious`: 添加虚假标注框
- `missing`: 删除标注框

**用法:**
```bash
python dataprocess/add_artificial_noise.py \
    --input annotations_val_coco.json \
    --output annotations_val_noisy.json \
    --noise-type label \
    --noise-ratio 0.2
```

## ⚙️ 配置说明

### 默认配置

所有实验脚本都使用以下默认配置：

- **模型**: `runs/detect/yolov8n_baseline_new/weights/best.pt`
- **数据集分割**: `val`（验证集）
- **噪声比例**: `0.2`（20%）
- **IoU阈值**: `0.5`

### 自定义配置

可以通过命令行参数自定义：

```bash
python experiments/experiment2_clod_effectiveness.py \
    --model <你的模型路径> \
    --split train \
    --noise-ratio 0.25
```

## 📝 注意事项

1. **运行顺序**: 建议按以下顺序运行：
   - 实验5（最简单，验证环境）
   - 实验2（核心实验）
   - 实验4（实用实验）
   - 实验6（IoU分析）
   - 实验3（需要ObjectLab）

2. **训练时间**: 
   - 实验4需要训练多个模型，可能需要较长时间
   - 实验2和6需要运行多次CLOD分析，也需要一定时间

3. **内存要求**:
   - 实验2、3、6会生成多个噪声数据集，注意磁盘空间
   - 建议至少10GB可用空间

4. **ObjectLab**:
   - 实验3需要ObjectLab实现，当前只运行CLOD部分
   - 如需完整对比，需要实现或安装ObjectLab

## 🐛 常见问题

### Q1: 找不到baseline模型

**A**: 先训练baseline模型：
```bash
python train_standard.py --data dataset.yaml --epochs 30 --name yolov8n_baseline_new
```

### Q2: SafeDNN-Clean脚本找不到

**A**: 确保 `otherwork/safednn-clean/safednn-clean.py` 存在

### Q3: 导入错误（scikit-learn）

**A**: 安装依赖：
```bash
pip install scikit-learn matplotlib pandas psutil
```

### Q4: 内存不足

**A**: 
- 减小batch size
- 使用`--workers 0`（单进程模式）
- 关闭其他程序释放内存

## 📚 参考

- [SafeDNN-Clean论文](https://arxiv.org/abs/2211.13993)
- [项目主README](../README.md)

---

**开始使用**: 建议从实验5开始，验证环境配置是否正确！ 🚀

