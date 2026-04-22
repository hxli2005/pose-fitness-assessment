#!/bin/bash
# 终止于任何报错
set -e

echo ">>> Creating virtual environment (.venv)..."
python3 -m venv .venv
source .venv/bin/activate

echo ">>> Updating pip..."
pip install --upgrade pip wheel && pip install "setuptools<60.0.0"

echo ">>> Installing PyTorch..."
# macOS 环境默认安装 CPU / MPS 版本的 PyTorch
pip install torch torchvision

echo ">>> Installing OpenMMLab ecosystem (mmpose, mmdet, mmcv)..."
pip install -U openmim
mim install mmengine
pip install "mmcv>=2.0.0" --no-build-isolation
mim install "mmdet>=3.0.0"
mim install "mmpose>=1.3.0"

echo ">>> Installing Ultralytics (YOLO11) and other utilities..."
pip install ultralytics pycocotools one-euro-filter seaborn matplotlib pandas jupyterlab

echo ">>> Verifying installation..."
python -c "import mmpose, ultralytics, pycocotools; print('✅ Core dependencies imported successfully!')"

echo ">>> Freezing requirements..."
pip freeze > requirements.txt

echo ">>> Setup complete! To activate the environment, run: source .venv/bin/activate"
