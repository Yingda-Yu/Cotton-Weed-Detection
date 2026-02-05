# Experiment Run Checklist

## Completed

### 1. Dataset
- [x] Dataset path set to `cotton weed dataset/`
- [x] VIA annotations converted to YOLO labels
- [x] Train: 593 label files; Val: 255 label files
- [x] Structure: `cotton weed dataset/{train,val}/{images,labels,annotations}`

### 2. Model
- [x] Baseline: `runs/detect/yolov8n_baseline_new/weights/best.pt`
- [x] mAP@0.5: 0.73065

### 3. Tools and dependencies
- [x] SafeDNN-Clean: `otherwork/safednn-clean/safednn-clean.py`
- [x] Installed: scikit-learn, matplotlib, pandas, psutil, cleanlab

### 4. Config
- [x] `dataset.yaml` and `dataset_cleaned.yaml` paths updated
- [x] Experiment script paths updated

## Experiments you can run

All 6 experiments are ready. Recommended order:

1. **Experiment 5: Manual Inspection** (easiest, sanity check)
2. **Experiment 2: CLOD Effectiveness**
3. **Experiment 4: Dataset Variants**
4. **Experiment 6: IoU Threshold Analysis**
5. **Experiment 1: Noise Impact**
6. **Experiment 3: CLOD vs SOTA**

(See `experiments/README.md` for exact commands.)

## Dataset stats

- **Train**: 593 images; carpetweed 446, morningglory 344, palmer_amaranth 271 (1061 annotations)
- **Val**: 255 images; carpetweed 156, morningglory 142, palmer_amaranth 173 (471 annotations)

## Notes

1. **Experiment 1** may need pre-built noisy datasets under `dataprocess/...`.
2. **Experiment 3** ObjectLab is not implemented; only CLOD is run.
3. **Experiment 4** trains multiple models and can take a long time.
4. Outputs go to `experiments/experimentX_results/`.

Ready to run experiments.
