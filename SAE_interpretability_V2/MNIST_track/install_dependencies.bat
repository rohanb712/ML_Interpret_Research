@echo off
REM Installation script for SAE Interpretability project
REM Run this in Command Prompt: install_dependencies.bat

echo Installing dependencies for SAE Interpretability...
echo.

set /p cuda="Do you have NVIDIA GPU with CUDA 12.1? (y/n): "

if /i "%cuda%"=="y" (
    echo.
    echo Installing PyTorch with CUDA 12.1...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo.
    echo Installing PyTorch (CPU-only)...
    pip install torch torchvision torchaudio
)

echo.
echo Installing other dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo Installing Tracr from GitHub...
pip install git+https://github.com/neelnanda-io/Tracr

echo.
echo Installation complete!
echo You can now run: python MNIST_SAE_jr.py
pause

