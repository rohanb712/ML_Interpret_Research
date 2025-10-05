# Installation script for SAE Interpretability project
# Run this in PowerShell: .\install_dependencies.ps1

Write-Host "Installing dependencies for SAE Interpretability..." -ForegroundColor Green

# Check if user has CUDA
$cuda = Read-Host "Do you have NVIDIA GPU with CUDA 12.1? (y/n)"

if ($cuda -eq "y" -or $cuda -eq "Y") {
    Write-Host "`nInstalling PyTorch with CUDA 12.1..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "`nInstalling PyTorch (CPU-only)..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio
}

Write-Host "`nInstalling other dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "`nInstalling Tracr from GitHub..." -ForegroundColor Yellow
pip install git+https://github.com/neelnanda-io/Tracr

Write-Host "`n✓ Installation complete!" -ForegroundColor Green
Write-Host "You can now run: python MNIST_SAE_jr.py" -ForegroundColor Cyan

