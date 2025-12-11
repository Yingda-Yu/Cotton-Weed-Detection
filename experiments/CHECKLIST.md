# 实验运行检查清单

## ✅ 已完成项目

### 1. 数据集准备
- [x] 数据集路径已更新为 `cotton weed dataset/`
- [x] VIA格式annotations已转换为YOLO格式labels
- [x] Train集：593个labels文件
- [x] Val集：255个labels文件
- [x] 数据集结构完整：
  ```
  cotton weed dataset/
  ├── train/
  │   ├── images/     ✅ 593张图片
  │   ├── labels/     ✅ 593个YOLO标签文件
  │   └── annotations/ ✅ 593个VIA格式文件（原始）
  └── val/
      ├── images/     ✅ 255张图片
      ├── labels/     ✅ 255个YOLO标签文件
      └── annotations/ ✅ 255个VIA格式文件（原始）
  ```

### 2. 模型准备
- [x] Baseline模型存在：`runs/detect/yolov8n_baseline_new/weights/best.pt`
- [x] 模型mAP@0.5: 0.73065

### 3. 工具和依赖
- [x] SafeDNN-Clean脚本：`otherwork/safednn-clean/safednn-clean.py`
- [x] Python依赖包已安装：
  - scikit-learn ✅
  - matplotlib ✅
  - pandas ✅
  - psutil ✅
  - cleanlab ✅

### 4. 配置文件
- [x] `dataset.yaml` 已更新路径
- [x] `dataset_cleaned.yaml` 已更新路径
- [x] 所有实验脚本路径已更新

## 🚀 现在可以运行的实验

所有6个实验现在都可以运行了！

### 推荐运行顺序

#### 1. 实验5: Manual Inspection（最简单，验证环境）
```bash
python experiments/experiment5_manual_inspection.py \
    --model runs/detect/yolov8n_baseline_new/weights/best.pt \
    --split train
```

#### 2. 实验2: CLOD Effectiveness（核心实验）
```bash
python experiments/experiment2_clod_effectiveness.py \
    --model runs/detect/yolov8n_baseline_new/weights/best.pt \
    --split val \
    --noise-ratio 0.2
```

#### 3. 实验4: Dataset Variants
```bash
python experiments/experiment4_dataset_variants.py \
    --model runs/detect/yolov8n_baseline_new/weights/best.pt \
    --split train \
    --suggestions-ratio 0.2
```

#### 4. 实验6: IoU Threshold Analysis
```bash
python experiments/experiment6_iou_threshold.py \
    --model runs/detect/yolov8n_baseline_new/weights/best.pt \
    --split val \
    --noise-ratio 0.2
```

#### 5. 实验1: Noise Impact
```bash
python experiments/experiment1_noise_impact.py
```

#### 6. 实验3: CLOD vs SOTA
```bash
python experiments/experiment3_clod_vs_sota.py \
    --model runs/detect/yolov8n_baseline_new/weights/best.pt \
    --split val \
    --noise-ratio 0.25
```

## 📊 数据集统计

### Train集
- 图片数：593
- 标注数：
  - carpetweed: 446
  - morningglory: 344
  - palmer_amaranth: 271
  - 总计：1061个标注

### Val集
- 图片数：255
- 标注数：
  - carpetweed: 156
  - morningglory: 142
  - palmer_amaranth: 173
  - 总计：471个标注

## ⚠️ 注意事项

1. **实验1**需要预先准备的噪声数据集（位于`dataprocess/cottonweed_split/train/noisy datasets/`）
2. **实验3**的ObjectLab部分需要单独实现（当前只运行CLOD部分）
3. **实验4**需要训练多个模型，可能需要较长时间
4. 所有实验的输出会保存在各自的`experiments/experimentX_results/`目录中

## 🎉 准备完成！

所有准备工作已完成，可以开始运行实验了！

