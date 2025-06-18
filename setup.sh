#!/bin/bash

echo "🚀 Setting up L2CS Gaze Tracking Environment..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "📦 Python version: $python_version"

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv l2cs_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source l2cs_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p captures
mkdir -p videos
mkdir -p pretrained_models

# Download L2CS model if not exists
echo "📥 Downloading L2CS model..."
python3 -c "
import os
from utils.model_utils import download_l2cs_pretrained_model
if not os.path.exists('pretrained_models/L2CSNet_gaze360.pkl'):
    download_l2cs_pretrained_model()
    print('✅ Model downloaded successfully!')
else:
    print('✅ Model already exists!')
"

echo "✅ Setup completed successfully!"
echo ""
echo "🎯 To run the gaze tracker:"
echo "   source l2cs_env/bin/activate"
echo "   ./run.sh"
echo ""
echo "🎯 To deactivate environment:"
echo "   deactivate" 