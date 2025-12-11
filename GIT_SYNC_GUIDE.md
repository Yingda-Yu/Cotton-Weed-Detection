# Git同步指南

本文档说明哪些文件会被上传到GitHub，哪些会被排除。

## ✅ 会被上传的文件

### 代码文件
- 所有Python脚本（`.py`）
- 配置文件（`.yaml`, `.yaml`）
- README和文档（`.md`）

### 实验结果（重要）
- `experiments/*_results/*_results.json` - 实验结果数据
- `experiments/*_results/*_report.json` - 实验报告
- `experiments/*_results/*_results.png` - 结果图表
- `experiments/*_results/*_analysis.png` - 分析图表
- `experiments/*_results/*_comparison.png` - 对比图表
- `experiments/*_results/*_distribution.png` - 分布图表
- `experiments/*_results/*_curves.png` - 曲线图
- `experiments/README.md` - 实验说明文档

### 工具和脚本
- `tools/` - 所有工具脚本
- `dataset/` - 数据集处理脚本
- `experiments/` - 所有实验脚本

## ❌ 会被排除的文件

### 大文件（超过GitHub限制）
- **数据集**：`cotton weed dataset/`（约几GB）
- **模型权重**：`*.pt`, `*.pth`（每个几MB到几百MB）
- **训练结果**：`runs/`（包含大量图片和权重）
- **实验训练结果**：`experiments/*/runs/`（模型权重和训练图片）

### 临时文件
- **输出文件**：`outputs/`（所有临时JSON文件）
- **质量报告**：`quality_report_*.json`（可重新生成）
- **标注文件**：`annotations_*_coco.json`（可重新生成）
- **预测文件**：`predictions_*_coco.json`（可重新生成）

### 缓存和临时数据
- `*.cache` - 缓存文件
- `labels.cache` - 标签缓存
- `__pycache__/` - Python缓存
- `*.log` - 日志文件

### 可视化样本
- `visualized_samples/` - 可视化样本图片
- `quality_issues/` - 质量问题可视化
- `experiments/**/train_batch*.jpg` - 训练批次图片
- `experiments/**/val_batch*.jpg` - 验证批次图片

## 📁 文件组织

### outputs/ 文件夹
所有临时生成的JSON文件已移动到 `outputs/` 文件夹：
- 质量报告文件
- COCO格式标注文件
- 预测结果文件
- 噪声标注文件
- 清洗后的标注文件

**这些文件会被.gitignore排除，不会上传到GitHub**

### experiments/ 文件夹
实验结果的重要文件会被保留：
- JSON报告文件（`*_results.json`, `*_report.json`）
- 重要图表（`*_results.png`, `*_analysis.png`等）
- 实验脚本（`.py`文件）
- README文档

大文件会被排除：
- 训练结果（`runs/`目录）
- 模型权重（`weights/`目录）
- 训练数据集副本（`train_*/`目录）
- 训练/验证批次图片

## 🔄 如何重新生成被排除的文件

如果需要这些文件，可以重新运行相应的脚本：

```bash
# 生成质量报告
python tools/run_label_quality_analysis.py --model <model_path> --split train

# 转换COCO格式
python dataset/yolo_to_coco.py --split train

# 生成预测
python dataset/generate_predictions_coco.py --model <model_path> --split train

# 运行实验
python experiments/experiment2_clod_effectiveness.py --model <model_path>
```

## 📊 预计上传大小

- **代码文件**：< 1 MB
- **实验结果JSON**：< 100 KB
- **实验结果图表**：< 5 MB
- **文档**：< 100 KB

**总计**：< 10 MB（符合GitHub要求）

## ⚠️ 注意事项

1. **数据集不会上传**：`cotton weed dataset/` 目录已被排除
2. **模型权重不会上传**：所有 `.pt` 文件已被排除
3. **临时文件不会上传**：`outputs/` 目录已被排除
4. **实验结果会被保留**：重要的JSON报告和图表会被上传

## 🚀 同步到GitHub

准备好后，可以执行：

```bash
git add .
git commit -m "Add experiment results and update documentation"
git push origin main
```

