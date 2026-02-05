# Git Sync Guide

This document describes which files are tracked by Git and which are ignored.

## Tracked files

### Code
- All Python scripts (`.py`)
- Config files (`.yaml`)
- README and docs (`.md`)

### Experiment results
- `experiments/*_results/*_results.json`
- `experiments/*_results/*_report.json`
- `experiments/*_results/*.png` (result/analysis/comparison/distribution/curves)
- `experiments/README.md`

### Tools and scripts
- `tools/`
- `dataset/`
- `experiments/` (scripts)

## Ignored / excluded

### Large files
- **Dataset**: `cotton weed dataset/` (multi-GB)
- **Weights**: `*.pt`, `*.pth`
- **Training outputs**: `runs/`
- **Experiment runs**: `experiments/*/runs/`

### Temporary / generated
- **Outputs**: `outputs/`
- **Quality reports**: `quality_report_*.json`
- **Annotations**: `annotations_*_coco.json`
- **Predictions**: `predictions_*_coco.json`
- `*.cache`, `labels.cache`, `__pycache__/`, `*.log`
- `visualized_samples/`, `quality_issues/`
- `experiments/**/train_batch*.jpg`, `experiments/**/val_batch*.jpg`

## Regenerating excluded files

```bash
python tools/run_label_quality_analysis.py --model <model_path> --split train
python dataset/yolo_to_coco.py --split train
python dataset/generate_predictions_coco.py --model <model_path> --split train
python experiments/experiment2_clod_effectiveness.py --model <model_path>
```

## Approximate upload size

- Code: &lt; 1 MB  
- Result JSONs: &lt; 100 KB  
- Result figures: &lt; 5 MB  
- Docs: &lt; 100 KB  

**Total**: &lt; 10 MB

## Notes

1. Dataset and model weights are not uploaded.
2. Temporary outputs under `outputs/` are ignored.
3. Important experiment JSON reports and figures are kept in the repo.

## Push to GitHub

```bash
git add .
git commit -m "Add experiment results and update documentation"
git push origin main
```
