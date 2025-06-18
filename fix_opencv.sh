#!/bin/bash

echo "🔧 Fixing OpenCV Installation Issues..."
echo "="*50

# Check if virtual environment exists
if [ ! -d "l2cs_env" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source l2cs_env/bin/activate

echo "📦 Current OpenCV packages:"
pip list | grep opencv

echo ""
echo "🧹 Removing conflicting OpenCV packages..."
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y

echo ""
echo "📥 Installing correct OpenCV package..."
pip install opencv-contrib-python>=4.8.0

echo ""
echo "✅ Verifying installation..."
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

echo ""
echo "🔍 Testing OpenCV functionality..."
python3 -c "
import cv2
import numpy as np

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
    print('⚠️  Camera access: Not available')

print('✅ OpenCV installation test completed!')
"

echo ""
echo "🎯 OpenCV fix completed!"
echo "You can now run the system check or gaze tracker." 