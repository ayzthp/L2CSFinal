#!/bin/bash

echo "🎯 Starting L2CS Gaze Tracker with Left Hand Focus Detection..."

# Check if virtual environment exists
if [ ! -d "l2cs_env" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source l2cs_env/bin/activate

# Check if main script exists
if [ ! -f "left_hand_focus_bisector_with_bbox.py" ]; then
    echo "❌ Main script not found: left_hand_focus_bisector_with_bbox.py"
    exit 1
fi

# Check if user wants to test cameras first
echo ""
echo "🔧 Options:"
echo "1. Run gaze tracker (camera 8)"
echo "2. Test available cameras"
echo "3. System check and monitoring"
echo "4. Exit"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "🚀 Launching gaze tracker on camera 8..."
        python3 left_hand_focus_bisector_with_bbox.py
        ;;
    2)
        echo "🔍 Testing available cameras..."
        python3 test_camera.py
        ;;
    3)
        echo "🔍 Running system check and monitoring..."
        python3 system_check.py
        ;;
    4)
        echo "👋 Goodbye!"
        ;;
    *)
        echo "❌ Invalid choice. Running gaze tracker..."
        python3 left_hand_focus_bisector_with_bbox.py
        ;;
esac

# Deactivate virtual environment on exit
deactivate 