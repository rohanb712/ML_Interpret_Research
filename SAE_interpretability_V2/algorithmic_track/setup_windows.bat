@echo off
REM Windows Setup Script for TRACR + TransformerLens + SAE

echo ============================================
echo TRACR TransformerLens SAE Setup for Windows
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.9 or later.
    pause
    exit /b 1
)

echo Step 1: Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo Step 4: Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo Step 5: Installing Tracr from GitHub...
pip install git+https://github.com/neelnanda-io/Tracr.git@main

echo.
echo ============================================
echo Installation Complete!
echo ============================================
echo.
echo To run the script:
echo   1. Activate the virtual environment: venv\Scripts\activate
echo   2. Run the script: python TRACR_TL_SAE.py
echo.
echo Note: This uses JAX CPU version. For GPU support, use WSL2.
echo ============================================
pause

