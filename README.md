# 🌾 Cotton Weed Detection

Data-centric AI for cotton weed detection using YOLOv8 and SafeDNN-Clean for automatic data cleaning and label quality improvement.

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Core Features](#core-features)
- [Data Cleaning Pipeline](#data-cleaning-pipeline)
- [Dataset Handling](#dataset-handling)
- [Usage Details](#usage-details)
- [FAQ](#faq)

---

## 🚀 Quick Start

### 1. Train a model

```bash
# Train on original data
python train_standard.py --data dataset.yaml --epochs 30

# Train on cleaned data
python train_standard.py --data dataset_cleaned.yaml --epochs 30
```

### 2. Generate predictions

```bash
python predict.py --model runs/detect/xxx/weights/best.pt
```

### 3. Full workflow (recommended)

```bash
# One-shot: baseline training → data cleaning → cleaned training → performance comparison
python run_complete_workflow.py
```

---

## 📁 Project Structure

```
Cotton Weed Detect/
├── train_standard.py          # Core: training script
├── predict.py                 # Core: prediction script
├── run_complete_workflow.py   # Core: full workflow
│
├── tools/                     # Utility scripts
│   ├── run_label_quality_analysis.py  # Label quality analysis
│   ├── clean_dataset.py               # Data cleaning
│   ├── run_cleaning_and_comparison.py # Cleaning & comparison pipeline
│   ├── visualize_quality_report.py   # Quality report visualization
│   └── visualize_annotations.py       # Annotation visualization
│
├── dataset/                   # Dataset utilities
│   ├── yolo_to_coco.py        # YOLO to COCO format
│   ├── coco_to_yolo.py        # COCO to YOLO format
│   └── generate_predictions_coco.py  # COCO-format predictions
│
├── dataset.yaml               # Dataset config (original)
├── dataset_cleaned.yaml       # Dataset config (cleaned)
│
├── train/                     # Training set
├── val/                       # Validation set
├── test/                      # Test set
├── cleaned_train/             # Cleaned training set
│
├── runs/                      # Training outputs
├── yolov8n.pt                 # Pretrained weights
│
└── otherwork/safednn-clean/   # SafeDNN-Clean framework
```

---

## 🎯 Core Features

### Training

**Basic usage:**
```bash
python train_standard.py --data dataset.yaml --epochs 30 --batch 8
```

**Arguments:**
- `--data`: Dataset YAML path (dataset.yaml or dataset_cleaned.yaml)
- `--epochs`: Number of epochs (default 30)
- `--batch`: Batch size (default 16)
- `--imgsz`: Image size (default 640)
- `--device`: Device (0 for GPU, 'cpu' for CPU)
- `--workers`: DataLoader workers (default 4; use 0 if low memory)
- `--name`: Run name (default yolov8n_standard)
- `--resume`: Path to checkpoint to resume training

**Example:**
```bash
# Resume from checkpoint
python train_standard.py \
    --data dataset_cleaned.yaml \
    --epochs 30 \
    --batch 4 \
    --workers 0 \
    --resume runs/detect/yolov8n_cleaned_train/weights/last.pt
```

### Prediction

**Basic usage:**
```bash
python predict.py --model runs/detect/xxx/weights/best.pt
```

**Arguments:**
- `--model`: Path to model weights
- `--conf`: Confidence threshold (default 0.25)
- `--output`: Output CSV path (default submission.csv)

---

## 🔧 Data Cleaning Pipeline

### Full workflow

Use `run_complete_workflow.py` to run all steps:

```bash
python run_complete_workflow.py
```

This runs:
1. Baseline model training
2. Training set label quality analysis
3. Training set annotation cleaning
4. Preparation of cleaned dataset
5. Training on cleaned data
6. Performance comparison

### Step-by-step

#### Step 1: Label quality analysis

```bash
python tools/run_label_quality_analysis.py \
    --model runs/detect/yolov8n_baseline/weights/best.pt \
    --split train
```

**Arguments:**
- `--model`: Model weights path
- `--split`: Split (train or val)
- `--iou`: IoU clustering threshold (default 0.5)
- `--threshold`: Quality score threshold (default 0.5)
- `--conf`: Prediction confidence threshold (default 0.25)

**Outputs:**
- `quality_report_train.json` – quality report
- `predictions_train_coco.json` – predictions

#### Step 2: Auto-clean data

```bash
python tools/clean_dataset.py \
    --quality-report quality_report_train.json \
    --predictions predictions_train_coco.json \
    --output cleaned_train_annotations.json \
    --location-threshold 0.7 \
    --label-threshold 0.8 \
    --missing-threshold 0.5
```

**Arguments:**
- `--quality-report`: Quality report path
- `--predictions`: Predictions path
- `--output`: Output path
- `--location-threshold`: Score threshold for Location fixes (default 0.7)
- `--label-threshold`: Score threshold for Label fixes (default 0.8)
- `--missing-threshold`: Score threshold for Missing additions (default 0.5)

**Cleaning rules:**
1. **Spurious**: Remove annotation
2. **Location**: Replace bbox with model prediction when score ≥ threshold
3. **Label**: Replace category with model prediction when score ≥ threshold
4. **Missing**: Add model prediction when score ≥ threshold

#### Step 3: Convert and prepare dataset

```bash
# Convert to YOLO format
python dataset/coco_to_yolo.py \
    --coco-file cleaned_train_annotations.json \
    --split train \
    --output-dir cleaned_train

# Copy images (Windows PowerShell)
Copy-Item -Path "train\images\*" -Destination "cleaned_train\images\" -Recurse
```

#### Step 4: Train on cleaned data

```bash
python train_standard.py --data dataset_cleaned.yaml --epochs 30
```

### Visualize quality report

```bash
python tools/visualize_quality_report.py \
    --report quality_report_train.json \
    --val-dir train \
    --output quality_issues \
    --top-n 50
```

---

## 📊 Dataset Handling

### Format conversion

#### YOLO to COCO

```bash
python dataset/yolo_to_coco.py --split train --output annotations_train_coco.json
```

#### COCO to YOLO

```bash
python dataset/coco_to_yolo.py \
    --coco-file cleaned_train_annotations.json \
    --split train \
    --output-dir cleaned_train
```

### Generate predictions

```bash
python dataset/generate_predictions_coco.py \
    --model runs/detect/xxx/weights/best.pt \
    --split train \
    --annotations annotations_train_coco.json \
    --output predictions_train_coco.json \
    --conf 0.25
```

---

## 📖 Usage Details

### Issue types (SafeDNN-Clean)

SafeDNN-Clean identifies four annotation issues:

1. **Spurious**
   - Annotated but not detected by model
   - Fix: Remove the annotation

2. **Missing**
   - Detected by model but not annotated
   - Fix: Add the missing annotation

3. **Location**
   - Correct class but wrong bbox
   - Fix: Adjust bbox (e.g. use model prediction)

4. **Label**
   - Wrong class
   - Fix: Correct the class label

### Quality report format

`quality_report_train.json` structure:

```json
{
  "annotations": [
    {
      "id": 123,
      "image_id": 45,
      "category_id": 0,
      "bbox": [100, 200, 50, 60],
      "quality": 0.32,
      "issue": "spurious"
    }
  ]
}
```

- **quality**: 0–1; lower means worse quality
- **issue**: spurious | missing | location | label

### Best practices

1. **Priority**: High-confidence false negatives (missing, conf > 0.7) → low-quality (quality < 0.3) → label → location
2. **Thresholds**: Conservative (high) = fix only high-confidence; aggressive (low) = fix more
3. **Iteration**: Start with conservative thresholds, then adjust and re-run

---

## 🐛 FAQ

### Q1: SafeDNN-Clean script not found?

**A:** Ensure `otherwork/safednn-clean/safednn-clean.py` exists.

### Q2: cleanlab import error?

**A:** Install cleanlab:
```bash
pip install cleanlab>=2.2.0
```

### Q3: Out of memory?

**A:**
- Use `--workers 0` (single process)
- Reduce `--batch`
- Increase Windows page file size if needed

### Q4: How to resume training?

**A:** Use `--resume`:
```bash
python train_standard.py --resume runs/detect/xxx/weights/last.pt
```

### Q5: Performance drops after cleaning?

**A:** Possible causes: too few epochs (try 30), too many removals, or thresholds need tuning.

---

## 📊 Experiment Results

Six experiments evaluate CLOD (SafeDNN-Clean) for data cleaning and label quality.

### Overview

| Experiment | Name | Main finding |
|------------|------|--------------|
| 1 | Noise impact | Effect of noise type on model performance |
| 2 | CLOD effectiveness | Label noise AUROC=0.79, location AUROC=0.86 |
| 3 | CLOD vs SOTA | CLOD label AUROC=0.88 at 25% noise |
| 4 | Dataset variants | **mAP@0.5 +50.39% after applying CLOD suggestions** |
| 5 | Manual inspection | 493 issues (40.5%), 264 spurious |
| 6 | IoU threshold | Best IoU=0.3, mean AUROC=0.67 |

### Experiment 2: CLOD effectiveness

**Setup:** 20% artificial noise on validation set (255 images, 471 annotations).

**Results (IoU=0.5):**

| Noise type | AUROC | Note |
|------------|-------|------|
| **Label** | **0.7876** | ✅ Strong |
| **Location_20** | **0.8571** | ✅ Strong |

**Conclusion:** CLOD works well for label and small location errors; limited for spurious and missing.

### Experiment 3: CLOD vs SOTA

**Setup:** 25% noise; compare CLOD with ObjectLab.

**Results:** CLOD label AUROC **0.8793**, location_20 **0.8571**; others similar to Exp. 2.

### Experiment 4: Dataset variants ⭐

**Setup:** Apply top 20% CLOD suggestions, then train and compare.

**Results:**

| Dataset | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---------|---------|---------------|-----------|--------|
| **Original** | 0.3710 | 0.1894 | 0.6260 | 0.5839 |
| **Suggestions** | **0.5579** | **0.4105** | **0.6708** | 0.5229 |
| **Change** | **+0.1869** | **+0.2211** | **+0.0448** | -0.0610 |
| **% change** | **+50.39%** | **+116.79%** | **+7.15%** | -10.45% |

- Original annotations: 1,061 → Suggestions: 724 (264 spurious removed); 98 suggestions applied (top 20%).

**Conclusion:** CLOD-based cleaning gave >50% mAP@0.5 gain, showing impact of data quality.

### Experiment 5: Manual inspection

**Setup:** CLOD on training set (593 images, 1,061 annotations).

**Results:** 493 annotations with quality scores (40.5%). Issue distribution: Spurious 264 (53.5%), Missing 156 (31.6%), Location 47 (9.5%), Label 26 (5.3%). Quality stats: min 0.04, max 0.50, mean 0.11.

### Experiment 6: IoU threshold

**Setup:** IoU thresholds 0.3–0.7.

**Results:** Best overall IoU=0.3 (mean AUROC=0.67). Location benefits from low IoU (0.3–0.4); Label/Scale from high (0.6–0.7).

### Summary

1. CLOD is strong on label and small location errors (AUROC>0.78).
2. Applying CLOD suggestions improved mAP@0.5 by 50.39%.
3. Spurious is the dominant issue type (53.5%).
4. Use IoU≈0.3 for general use; tune by noise type.

All experiment outputs are under `experiments/` (JSON reports and figures).

---

## 📚 Dataset Info

- **Task:** Multi-class weed detection (3 classes)
- **Format:** YOLO (normalized coordinates)
- **Model:** YOLOv8n (fixed for competition)
- **Train:** 593 images, 1,061 annotations
- **Val:** 255 images, 471 annotations
- **Test:** 170 images (no labels)

### Classes

- **0:** Carpetweed
- **1:** Morning Glory
- **2:** Palmer Amaranth

---

## 📝 Submission Format

CSV columns: `image_id`, `prediction_string`

**Prediction string format:**
```
class_id confidence x_center y_center width height
```

**Example:**
```csv
image_id,prediction_string
20190613_6062W_CM_29,0 0.95 0.5 0.3 0.2 0.4 1 0.87 0.7 0.6 0.15 0.25
20200624_iPhone6_SY_132,no box
```

**Rules:** Column names lowercase; coordinates in [0, 1]; use `no box` when no detections.

---

## 🔗 References

- [SafeDNN-Clean paper](https://arxiv.org/abs/2211.13993)
- [CleanLab docs](https://docs.cleanlab.ai/)
- [YOLOv8 docs](https://docs.ultralytics.com/)
- [COCO format](https://cocodataset.org/#format-data)

---

## 📄 License

Dataset for competition use only; see official competition rules.

---

**Get started:** `python run_complete_workflow.py` 🚀
