@echo off
echo ========================================
echo  Egyptian Prescription OCR - Setup
echo ========================================

echo.
echo [1/3] Creating conda environment...
call conda env create -f environment.yml
if errorlevel 1 (
    echo ERROR: Failed to create conda environment.
    pause
    exit /b 1
)

echo.
echo [2/3] Installing PyTorch...
call conda activate ocr310
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo CUDA version failed, installing CPU version instead...
    pip install torch==2.5.1 torchvision==0.20.1
)

echo.
echo [3/3] Installing other packages...
pip install -r requirements.txt

echo.
echo ========================================
echo  Environment ready!
echo  Next step: run download_models.bat
echo  to download the model weights.
echo ========================================
pause