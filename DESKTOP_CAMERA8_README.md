# Desktop Camera 8 L2CS Gaze Tracker

Dedicated setup and run scripts for desktop use with camera 8.

## 🎯 Purpose

This setup is specifically designed for desktop environments that require camera 8 for gaze tracking with left hand focus detection.

## 🚀 Quick Start Commands

### 1. **Setup Desktop Environment**
```bash
./desktop_setup.sh
```

### 2. **Run Desktop Camera 8 Tracker**
```bash
./desktop_run.sh
```

## 📋 Prerequisites

- **Camera 8**: External camera connected and available
- **Python 3.8+**: Installed on your system
- **Permissions**: Camera access granted to the application

## 🔧 Setup Process

The `desktop_setup.sh` script will:

1. ✅ Create dedicated virtual environment (`desktop_l2cs_env`)
2. ✅ Install OpenCV with camera 8 support
3. ✅ Install all required dependencies
4. ✅ Download L2CS model
5. ✅ Test camera 8 availability
6. ✅ Verify OpenCV installation

## 🎮 Usage

### **Running the Tracker**
```bash
./desktop_run.sh
```

### **Controls**
- **Q**: Quit the application
- **S**: Save current frame with metadata
- **Look at your left hand**: The system will detect focus

### **Visual Indicators**
- **Blue circles**: Pupils and nose tip
- **Yellow bounding box**: Left hand detection (not focused)
- **Green bounding box**: Left hand detection (focused)
- **Blue arrow**: Gaze direction
- **Yellow arrow**: Left-looking detection
- **Green lines**: Focus confirmation (pupils to hand)

## 🔍 Camera 8 Requirements

### **Hardware**
- External camera connected to camera 8
- USB camera or webcam
- Minimum resolution: 640x480
- Recommended FPS: 30

### **Software**
- Camera drivers installed
- Camera permissions granted
- No other application using camera 8

## 🐛 Troubleshooting

### **Camera 8 Not Available**
```bash
# Check camera availability
python3 test_camera.py

# Test OpenCV installation
python3 test_opencv.py
```

### **Common Issues**

1. **"Camera 8 is not available"**
   - Connect external camera
   - Check camera permissions
   - Restart application

2. **"OpenCV installation failed"**
   ```bash
   ./fix_opencv.sh
   ```

3. **"Virtual environment not found"**
   ```bash
   ./desktop_setup.sh
   ```

## 📁 File Structure

```
L2CS-final/
├── desktop_setup.sh                    # Desktop environment setup
├── desktop_run.sh                      # Desktop camera 8 launcher
├── desktop_camera8_gaze_tracker.py     # Main desktop tracker
├── desktop_l2cs_env/                   # Desktop virtual environment
├── captures/                           # Saved frames
└── [other files...]
```

## ⚡ Performance Features

- **Smooth left-looking detection**: Optimized for responsive left-looking detection
- **30 FPS target**: Smooth real-time performance
- **Camera 8 optimization**: Dedicated for desktop camera 8
- **Focus bonus system**: Additional focus points when looking left
- **Gradual focus decay**: Prevents sudden focus loss

## 🔄 Alternative Options

If camera 8 is not available, you can use the regular tracker with camera fallback:

```bash
./run.sh
```

This will try camera 8 first, then fallback to camera 0.

## 📞 Support

For issues with desktop camera 8 setup:

1. Check camera connections
2. Verify camera permissions
3. Run system check: `python3 system_check.py`
4. Test OpenCV: `python3 test_opencv.py`

---

**Happy Desktop Gaze Tracking! 👁️✋📷** 