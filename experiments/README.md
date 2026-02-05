# Experiment Scripts

This directory contains 6 experiment scripts to reproduce the paper experiments.

## Experiment list

| Experiment | Script | Description |
|------------|--------|-------------|
| 1 | `experiment1_noise_impact.py` | Noise impact analysis |
| 2 | `experiment2_clod_effectiveness.py` | CLOD effectiveness |
| 3 | `experiment3_clod_vs_sota.py` | CLOD vs SOTA (CLOD only) |
| 4 | `experiment4_dataset_variants.py` | Dataset variants |
| 5 | `experiment5_manual_inspection.py` | Manual inspection |
| 6 | `experiment6_iou_threshold.py` | IoU threshold analysis |

## Quick start

### Prerequisites

1. **Trained baseline model**
   ```bash
   python train_standard.py --data dataset.yaml --epochs 30 --name yolov8n_baseline_new
   ```

2. **Dependencies**
   ```bash
   pip install scikit-learn matplotlib pandas psutil
   ```

3. **SafeDNN-Clean**: ensure `otherwork/safednn-clean/safednn-clean.py` exists.

### Run experiments

#### Experiment 5: Manual Inspection (run first)

```bash
python experiments/experiment5_manual_inspection.py \
    --model runs/detect/yolov8n_baseline_new/weights/best.pt \
    --split train
```

Outputs: `experiment5_results/manual_inspection_report.json`, `quality_distribution.png`

#### Experiment 2: CLOD Effectiveness

```bash
python experiments/experiment2_clod_effectiveness.py \
    --model runs/detect/yolov8n_baseline_new/weights/best.pt \
    --split val --noise-ratio 0.2
```

Adds 20% artificial noise; evaluates label, location, scale, spurious, missing; outputs AUROC and ROC.

#### Experiment 4: Dataset Variants

```bash
python experiments/experiment4_dataset_variants.py \
    --model runs/detect/yolov8n_baseline_new/weights/best.pt \
    --split train --suggestions-ratio 0.2
```

Builds Suggestions dataset (top 20% CLOD suggestions), trains and compares.

#### Experiment 6: IoU Threshold

```bash
python experiments/experiment6_iou_threshold.py \
    --model runs/detect/yolov8n_baseline_new/weights/best.pt \
    --split val --noise-ratio 0.2
```

Sweeps IoU 0.3–0.7 and reports best IoU.

#### Experiment 3: CLOD vs SOTA

```bash
python experiments/experiment3_clod_vs_sota.py \
    --model runs/detect/yolov8n_baseline_new/weights/best.pt \
    --split val --noise-ratio 0.25
```

Note: ObjectLab is not implemented; only CLOD is run.

#### Experiment 1: Noise Impact

```bash
python experiments/experiment1_noise_impact.py
```

Trains on noisy datasets and plots noise type vs mAP@0.5.

## Output layout

```
experiments/
├── experiment2_results/   (JSON, PNG)
├── experiment3_results/   (JSON, PNG, MD)
├── experiment4_results/   (report, runs/)
├── experiment5_results/   (report, PNG)
└── experiment6_results/   (JSON, PNG)
```

## Artificial noise module

`dataprocess/add_artificial_noise.py` adds artificial noise to COCO annotations.

Noise types: `label`, `location`, `scale`, `spurious`, `missing`. Example:

```bash
python dataprocess/add_artificial_noise.py \
    --input annotations_val_coco.json --output annotations_val_noisy.json \
    --noise-type label --noise-ratio 0.2
```

## Defaults

- Model: `runs/detect/yolov8n_baseline_new/weights/best.pt`
- Split: `val`
- Noise ratio: `0.2`
- IoU: `0.5`

Override via CLI (e.g. `--model <path> --split train --noise-ratio 0.25`).

## Notes

1. Run order: 5 → 2 → 4 → 6 → 3 (5 first to verify env).
2. Experiment 4 can take a long time (multiple trainings).
3. Experiments 2, 3, 6 use disk for noisy datasets; have ~10GB free.
4. Experiment 3: only CLOD is run; ObjectLab must be implemented separately.

## FAQ

- **No baseline**: `python train_standard.py --data dataset.yaml --epochs 30 --name yolov8n_baseline_new`
- **SafeDNN-Clean missing**: ensure `otherwork/safednn-clean/safednn-clean.py` exists
- **Import error**: `pip install scikit-learn matplotlib pandas psutil`
- **OOM**: reduce batch, use `--workers 0`, close other apps

## References

- [SafeDNN-Clean paper](https://arxiv.org/abs/2211.13993)
- [Main README](../README.md)

Start with Experiment 5 to verify the environment.
