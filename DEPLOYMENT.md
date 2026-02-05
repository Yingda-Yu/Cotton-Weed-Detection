# Remote Server Deployment Guide

This guide explains how to deploy the Cotton Weed Detection project environment on a remote server using miniconda.

## Prerequisites

- Miniconda or Anaconda installed on the server
- GPU (recommended) or CPU
- Network access for installing dependencies

## Deployment Steps

### Method 1: Using conda environment.yml (recommended)

#### Step 1: Upload files to the server

Upload to the project directory on the server:
- `environment.yml` – conda environment config
- Project source code

#### Step 2: Create conda environment

```bash
cd /path/to/Cotton-Weed-Detect
conda env create -f environment.yml
conda activate cotton-weed-detect
```

#### Step 3: Verify installation

```bash
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLO installed successfully')"
```

### Method 2: Using pip requirements.txt

If conda fails, use pip:

```bash
conda create -n cotton-weed-detect python=3.9 -y
conda activate cotton-weed-detect

# PyTorch – choose according to CUDA version
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Or CPU only
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

pip install -r requirements.txt
```

## Configuration

### CUDA version

If the server CUDA version is not 11.8, adjust `environment.yml` (e.g. `pytorch-cuda=12.1`) or install PyTorch via pip:

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Verify GPU

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('CUDA version:', getattr(torch.version, 'cuda', 'N/A')); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
```

## Export local environment (optional)

```bash
conda activate cotton-weed-detect
conda env export --no-builds > environment.yml
# or: pip freeze > requirements.txt
```

Exported files may contain machine-specific paths; prefer the project’s provided configs.

## Test deployment

```bash
python train_standard.py --data dataset.yaml --epochs 1 --batch 2
python predict.py --model runs/detect/xxx/weights/best.pt
```

## Troubleshooting

### Q1: Conda env create fails

- Check network
- Try mirror channels if needed (e.g. Tsinghua)

### Q2: CUDA version mismatch

- Check CUDA: `nvidia-smi`
- Adjust `environment.yml` or install matching PyTorch via pip

### Q3: Out of memory

- Use `--batch 4` or `--batch 2`, `--workers 0`, or `--device cpu`

### Q4: Dependency conflicts

- Use a fresh conda env; install PyTorch first, then the rest; or install with pip one by one

## Files

- **environment.yml**: Conda env with dependencies and versions
- **requirements.txt**: Pip fallback
- **DEPLOYMENT.md**: This guide

## Update env

```bash
conda env update -f environment.yml --prune
# or: pip install -r requirements.txt --upgrade
```

## Best practices

1. Always use the conda env to avoid polluting system Python.
2. Pin versions in production for reproducibility.
3. Verify GPU after deployment.
4. Run a short training/inference test after setup.

---

After deployment you can run the project on the remote server.
