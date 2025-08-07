#!/bin/bash

# AV-HuBERT Refactored Package Installation Script
# This script helps you install the refactored AV-HuBERT package

set -e  # Exit on any error

echo "🚀 Installing AV-HuBERT Refactored Package"
echo "=========================================="

# Check if Python 3.8+ is available
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv avhubert_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source avhubert_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version by default)
echo "🔥 Installing PyTorch..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "📚 Installing other dependencies..."
pip install transformers datasets accelerate

# Install audio and video processing libraries
echo "🎵 Installing audio/video processing libraries..."
pip install librosa soundfile opencv-python

# Install other utilities
echo "🔧 Installing utilities..."
pip install numpy scipy omegaconf hydra-core tqdm matplotlib seaborn

# Install development dependencies
echo "🛠️  Installing development dependencies..."
pip install black isort flake8 pytest

# Install the package in development mode
echo "📦 Installing AV-HuBERT package..."
pip install -e .

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "  source avhubert_env/bin/activate"
echo ""
echo "To test the installation, run:"
echo "  python test_basic.py"
echo ""
echo "To run examples, run:"
echo "  python examples/basic_usage.py"
echo ""
echo "Happy coding! 🚀"