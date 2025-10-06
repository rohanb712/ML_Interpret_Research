#!/bin/bash
# Linux/Mac Setup Script for TRACR + TransformerLens + SAE

echo "============================================"
echo "TRACR TransformerLens SAE Setup"
echo "============================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found! Please install Python 3.9 or later."
    exit 1
fi

echo "Step 1: Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo "Step 2: Activating virtual environment..."
source venv/bin/activate

echo "Step 3: Upgrading pip..."
pip install --upgrade pip

echo "Step 4: Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Step 5: Installing Tracr from GitHub..."
pip install git+https://github.com/neelnanda-io/Tracr.git@main

echo
echo "============================================"
echo "Installation Complete!"
echo "============================================"
echo
echo "To run the script:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the script: python TRACR_TL_SAE.py"
echo
echo "Note: This uses JAX CPU version by default."
echo "============================================"

