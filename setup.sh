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

# Install OpenCV first to avoid conflicts
echo "📦 Installing OpenCV (fixing conflicts)..."
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y 2>/dev/null || true
pip install opencv-contrib-python>=4.8.0

# Install other requirements
echo "📦 Installing other dependencies..."
pip install torch>=1.9.0 torchvision>=0.10.0
pip install mediapipe>=0.8.0
pip install numpy>=1.21.0 scipy>=1.7.0
pip install matplotlib>=3.4.0 Pillow>=8.3.0
pip install gdown>=4.4.0 onnx>=1.10.0 onnxruntime>=1.8.0
pip install tqdm>=4.62.0 scikit-learn>=1.0.0 psutil>=5.8.0

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

# Test OpenCV installation
echo "🔍 Testing OpenCV installation..."
python3 -c "
import cv2
import numpy as np
print(f'✅ OpenCV version: {cv2.__version__}')

# Test basic functionality
img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(img, (10, 10), (90, 90), (0, 255, 0), 2)
print('✅ OpenCV basic functionality: OK')

# Test camera functionality
cap = cv2.VideoCapture(8)
if cap.isOpened():
    print('✅ Camera access: OK')
    cap.release()
else:
    print('⚠️  Camera access: Not available (this is normal if no camera connected)')

print('✅ OpenCV installation test completed!')
"

echo "✅ Setup completed successfully!"
echo ""
echo "🎯 To run the gaze tracker:"
echo "   source l2cs_env/bin/activate"
echo "   ./run.sh"
echo ""
echo "🎯 If you have OpenCV issues later:"
echo "   ./fix_opencv.sh"
echo ""
echo "🎯 To deactivate environment:"
echo "   deactivate" 