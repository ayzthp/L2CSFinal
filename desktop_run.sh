#!/bin/bash

echo "🎯 Starting Desktop Camera 8 L2CS Gaze Tracker..."
echo "="*50

# Check if virtual environment exists
if [ ! -d "desktop_l2cs_env" ]; then
    echo "❌ Desktop virtual environment not found."
    echo "🔧 Please run desktop_setup.sh first:"
    echo "   ./desktop_setup.sh"
    exit 1
fi

# Activate virtual environment
source desktop_l2cs_env/bin/activate

# Check if main script exists
if [ ! -f "desktop_camera8_gaze_tracker.py" ]; then
    echo "❌ Desktop camera 8 script not found: desktop_camera8_gaze_tracker.py"
    exit 1
fi

# Check camera 8 availability first
echo "🔍 Checking Camera 8 availability..."
python3 -c "
import cv2
cap = cv2.VideoCapture(8)
if cap.isOpened():
    print('✅ Camera 8 is available!')
    cap.release()
else:
    print('❌ Camera 8 is not available!')
    print('💡 Please ensure:')
    print('   - External camera is connected')
    print('   - Camera permissions are granted')
    print('   - No other application is using camera 8')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "🔧 Camera 8 is not available. Please:"
    echo "   1. Connect an external camera"
    echo "   2. Check camera permissions"
    echo "   3. Restart the application"
    echo ""
    echo "💡 Alternative: Use the regular tracker with camera fallback:"
    echo "   ./run.sh"
    exit 1
fi

echo ""
echo "🚀 Launching Desktop Camera 8 Gaze Tracker..."
echo "📷 Using Camera 8 (Desktop)"
echo "👁️ Optimized for smooth left-looking detection"
echo ""

# Run the desktop camera 8 gaze tracker
python3 desktop_camera8_gaze_tracker.py

# Deactivate virtual environment on exit
deactivate 