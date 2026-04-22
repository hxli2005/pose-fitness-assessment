#!/bin/bash
set -e

# 设置根目录到项目根
PROJECT_ROOT=$(cd $(dirname $0)/.. && pwd)
DATA_DIR="${PROJECT_ROOT}/data/coco"

echo ">>> Creating data directories..."
mkdir -p "${DATA_DIR}"

echo ">>> Navigating to data directory..."
cd "${DATA_DIR}"

# 1. 下载图片 (如果没下载)
if [ ! -d "val2017" ]; then
    echo ">>> Downloading COCO val2017 images (778 MB)..."
    curl -O "http://images.cocodataset.org/zips/val2017.zip"
    echo ">>> Extracting images..."
    unzip -q val2017.zip
    rm val2017.zip
else
    echo ">>> COCO val2017 images already exist. Skipping download."
fi

# 2. 下载标注 (如果没下载)
if [ ! -d "annotations" ]; then
    echo ">>> Downloading COCO val2017 annotations (241 MB)..."
    curl -O "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    echo ">>> Extracting annotations..."
    unzip -q annotations_trainval2017.zip
    rm annotations_trainval2017.zip
else
    echo ">>> COCO val2017 annotations already exist. Skipping download."
fi

echo ">>> Data preparation complete!"
echo "Images path: ${DATA_DIR}/val2017"
echo "Annotations path: ${DATA_DIR}/annotations/person_keypoints_val2017.json"
