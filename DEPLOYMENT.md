# 远程服务器部署指南

本指南说明如何在远程服务器上使用miniconda部署Cotton Weed Detection项目环境。

## 📋 前置要求

- 远程服务器已安装miniconda或anaconda
- 服务器有GPU（推荐）或CPU
- 网络连接正常（用于下载依赖）

## 🚀 部署步骤

### 方法1：使用conda environment.yml（推荐）

#### 步骤1：上传文件到服务器

将以下文件上传到服务器项目目录：
- `environment.yml` - conda环境配置文件
- 项目代码文件

#### 步骤2：创建conda环境

```bash
# 进入项目目录
cd /path/to/Cotton-Weed-Detect

# 使用environment.yml创建环境
conda env create -f environment.yml

# 激活环境
conda activate cotton-weed-detect
```

#### 步骤3：验证安装

```bash
# 检查Python版本
python --version

# 检查PyTorch和CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查YOLO
python -c "from ultralytics import YOLO; print('YOLO installed successfully')"
```

### 方法2：使用pip requirements.txt

如果conda环境创建失败，可以使用pip方式：

```bash
# 创建新的conda环境（仅Python）
conda create -n cotton-weed-detect python=3.9 -y
conda activate cotton-weed-detect

# 安装PyTorch（根据服务器CUDA版本选择）
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 或CPU版本
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 安装其他依赖
pip install -r requirements.txt
```

## 🔧 配置说明

### CUDA版本调整

如果服务器的CUDA版本不是11.8，需要修改`environment.yml`：

```yaml
# 对于CUDA 12.1
cudatoolkit=12.1

# 对于CPU版本，删除cudatoolkit行
```

或者使用pip安装PyTorch：

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 验证GPU可用性

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
```

## 📝 从本地环境导出（可选）

如果你想从本地已配置好的环境导出：

```bash
# 激活本地环境
conda activate "cotton weed detect"

# 导出环境（不包含构建信息，更通用）
conda env export --no-builds > environment.yml

# 或导出为requirements.txt格式
pip freeze > requirements.txt
```

**注意**：导出的文件可能包含本地特定的路径，建议使用本项目提供的通用版本。

## ✅ 测试部署

部署完成后，运行以下命令测试：

```bash
# 测试训练脚本
python train_standard.py --data dataset.yaml --epochs 1 --batch 2

# 测试预测脚本（需要先有训练好的模型）
python predict.py --model runs/detect/xxx/weights/best.pt
```

## 🐛 常见问题

### Q1: conda环境创建失败

**解决方案**：
- 检查网络连接
- 尝试使用国内镜像源：
  ```bash
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  conda config --set show_channel_urls yes
  ```

### Q2: CUDA版本不匹配

**解决方案**：
- 检查服务器CUDA版本：`nvidia-smi`
- 修改`environment.yml`中的`cudatoolkit`版本
- 或使用pip安装对应版本的PyTorch

### Q3: 内存不足

**解决方案**：
- 减小batch size：`--batch 4` 或 `--batch 2`
- 设置workers为0：`--workers 0`
- 使用CPU训练：`--device cpu`

### Q4: 依赖冲突

**解决方案**：
- 使用全新的conda环境
- 先安装PyTorch，再安装其他依赖
- 如果仍有问题，使用`pip install`逐个安装

## 📦 环境文件说明

- **environment.yml**: conda环境配置文件，包含所有依赖和版本信息
- **requirements.txt**: pip依赖文件，作为备选方案
- **DEPLOYMENT.md**: 本部署指南

## 🔄 更新环境

如果项目依赖有更新：

```bash
# 更新conda环境
conda env update -f environment.yml --prune

# 或更新pip依赖
pip install -r requirements.txt --upgrade
```

## 💡 最佳实践

1. **使用虚拟环境**：始终在conda环境中工作，避免污染系统Python
2. **固定版本**：生产环境建议固定依赖版本，确保可复现性
3. **GPU检查**：部署后立即验证GPU是否可用
4. **测试运行**：部署完成后运行小规模测试，确保一切正常

---

**部署完成后，你就可以在远程服务器上运行项目了！** 🎉
