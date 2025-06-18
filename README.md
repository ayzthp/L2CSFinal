# L2CS Gaze Tracking with Left Hand Focus Detection

A real-time gaze tracking system using L2CS-Net with MediaPipe integration for detecting when a user is focusing on their left hand. Optimized for smooth left-looking detection with reduced delay.

## 🚀 Features

- **Real-time Gaze Tracking**: Uses L2CS-Net for accurate gaze estimation
- **Left Hand Detection**: MediaPipe Hands for precise hand tracking with bounding box
- **Focus Detection**: Calculates similarity between gaze vector and hand position
- **Smooth Left-Looking Detection**: Optimized with smoothing filters for responsive left-looking detection
- **Visual Feedback**: Real-time bounding boxes, arrows, and status indicators
- **Posture Alignment**: Checks if user is properly aligned when focusing
- **Frame Capture**: Save focused frames with metadata

## 🛠️ Quick Setup

### Prerequisites
- Python 3.8 or higher
- Webcam (configured for camera 8 on desktop)
- macOS/Linux/Windows

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd L2CS-final
   ```

2. **Run the setup script**:
   ```bash
   ./setup.sh
   ```
   
   This will:
   - Create a virtual environment (`l2cs_env`)
   - Install all dependencies
   - Download the L2CS model
   - Create necessary directories

3. **Run the gaze tracker**:
   ```bash
   ./run.sh
   ```
   
   The run script will give you options:
   - **Option 1**: Run gaze tracker on camera 8 (default for desktop)
   - **Option 2**: Test available cameras
   - **Option 3**: Exit

### Camera Configuration
- **Default**: Camera 8 (configured for desktop)
- **Testing**: Use the camera test utility to find available cameras
- **Custom**: Modify `camera_id` parameter in the main script

## 🎯 Usage

### Controls
- **Q**: Quit the application
- **S**: Save current frame with metadata
- **Look at your left hand**: The system will detect focus and show visual indicators

### Visual Indicators
- **Blue circles**: Pupils and nose tip
- **Yellow bounding box**: Left hand detection (not focused)
- **Green bounding box**: Left hand detection (focused)
- **Blue arrow**: Gaze direction
- **Yellow arrow**: Left-looking detection
- **Green lines**: Focus confirmation (pupils to hand)
- **Status text**: Real-time feedback

### Focus Detection
The system detects focus when:
1. Gaze vector strength is above threshold
2. Similarity between gaze and hand direction is high
3. User maintains focus for several frames
4. **Bonus**: Left-looking detection provides additional focus points

## 📁 Project Structure

```
L2CS-final/
├── setup.sh                          # Environment setup script
├── run.sh                            # Application launcher
├── requirements.txt                  # Python dependencies
├── left_hand_focus_bisector_with_bbox.py  # Main application
├── l2cs/                            # L2CS model implementation
├── utils/                           # Utility functions
├── pretrained_models/               # Downloaded models
├── captures/                        # Saved frames
└── videos/                          # Video files
```

## 🔧 Technical Details

### Gaze Smoothing
- **SmoothGazeFilter**: Reduces jitter and improves left-looking detection
- **Exponential smoothing**: α=0.8 for immediate response
- **Moving average**: Window size=3 for stability
- **Blended approach**: Combines both methods for optimal performance

### Left-Looking Detection
- **Threshold**: -0.3 for left-looking detection
- **Counter**: Tracks consecutive left-looking frames
- **Bonus system**: Provides focus bonus when looking left
- **Visual feedback**: Yellow arrows and text indicators

### Performance Optimizations
- **30 FPS target**: Optimized for smooth real-time performance
- **Gradual focus decay**: Prevents sudden focus loss
- **Efficient processing**: Minimal delay in gaze vector updates
- **GPU acceleration**: Automatic CUDA detection

## 🐛 Troubleshooting

### Common Issues

1. **"Virtual environment not found"**
   ```bash
   ./setup.sh
   ```

2. **"Model not found"**
   - The setup script should download it automatically
   - Check internet connection
   - Run setup again

3. **Poor performance**
   - Ensure good lighting
   - Position face clearly in camera
   - Check if GPU is available

4. **Hand not detected**
   - Ensure left hand is visible
   - Good lighting conditions
   - Hand should be open and facing camera

### Performance Tips
- **Lighting**: Ensure good, even lighting on face and hands
- **Distance**: Position 30-60cm from camera
- **Stability**: Minimize head movement for better gaze tracking
- **Hand position**: Keep left hand clearly visible and open

## 📊 Metrics

The system tracks:
- **FPS**: Real-time performance
- **Focus counter**: Consecutive focused frames
- **Left-looking frames**: Total left-looking detection
- **Alignment score**: Posture quality
- **Gaze strength**: Confidence in gaze estimation

## 🔄 Development

### Adding New Features
1. Modify `left_hand_focus_bisector_with_bbox.py`
2. Test with `./run.sh`
3. Update documentation

### Customization
- **Focus threshold**: Adjust `focus_threshold` in constructor
- **Smoothing**: Modify `SmoothGazeFilter` parameters
- **Left-looking threshold**: Change `left_looking_threshold`
- **Visual elements**: Customize colors and indicators

## 📝 License

This project is based on L2CS-Net and MediaPipe. Please refer to their respective licenses.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue with detailed description

---

**Happy Gaze Tracking! 👁️✋** 