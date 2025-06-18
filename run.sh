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

# Run the gaze tracker
echo "🚀 Launching gaze tracker..."
python3 left_hand_focus_bisector_with_bbox.py

# Deactivate virtual environment on exit
deactivate 