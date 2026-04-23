#!/bin/bash
echo "========================================"
echo " Egyptian Prescription OCR - Setup"
echo "========================================"

echo ""
echo "[1/3] Creating conda environment..."
conda env create -f environment.yml
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create conda environment."
    exit 1
fi

echo ""
echo "[2/3] Installing PyTorch..."
conda run -n ocr310 pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
if [ $? -ne 0 ]; then
    echo "CUDA version failed, installing CPU version instead..."
    conda run -n ocr310 pip install torch==2.5.1 torchvision==0.20.1
fi

echo ""
echo "[3/3] Installing other packages..."
conda run -n ocr310 pip install -r requirements.txt

echo ""
echo "========================================"
echo " Environment ready!"
echo " Next step: run bash download_models.sh"
echo "========================================"