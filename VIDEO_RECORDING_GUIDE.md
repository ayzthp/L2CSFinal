# Video Recording Guide

## Overview
The `left_hand_focus_bisector_video.py` script provides video recording functionality for the left hand focus bisector tracker. You can record your gaze tracking sessions and save them as MP4 videos.

## Features

### Video Recording
- **Start/Stop Recording**: Press `'r'` to toggle video recording
- **Automatic Naming**: Videos are saved with timestamps (e.g., `left_hand_bisector_session_1703123456.mp4`)
- **Visual Indicators**: Red "RECORDING" text and circle indicator when recording
- **Status Check**: Press `'v'` to check current recording status

### File Management
- **Video Directory**: All videos are saved in the `videos/` folder
- **Frame Saving**: Press `'s'` to save individual frames as images
- **Automatic Cleanup**: Recording stops automatically when you quit

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `s` | Save current frame as image |
| `r` | Start/Stop video recording |
| `v` | View recording status |

## Usage

1. **Run the script**:
   ```bash
   python left_hand_focus_bisector_video.py
   ```

2. **Start recording**:
   - Press `'r'` to start recording
   - You'll see "RECORDING" text and a red circle indicator
   - Look at your left hand to activate focus detection

3. **Stop recording**:
   - Press `'r'` again to stop recording
   - The video will be saved automatically

4. **Check status**:
   - Press `'v'` to see current recording status in the terminal

## Video Specifications

- **Format**: MP4
- **Codec**: MP4V
- **Frame Rate**: 30 FPS
- **Resolution**: Same as camera input (typically 640x480)
- **Quality**: Full quality, no compression

## File Structure

```
L2CS final/
├── videos/                          # Video recordings
│   ├── left_hand_bisector_session_1703123456.mp4
│   └── ...
├── captures/                        # Individual frame captures
│   ├── left_bisector_focused_aligned_1703123456.jpg
│   └── ...
└── left_hand_focus_bisector_video.py
```

## Testing

To test the video recording functionality independently:

```bash
python test_video_recording.py
```

This will create a simple test recording to verify that video recording works correctly on your system.

## Troubleshooting

### Common Issues

1. **No video file created**:
   - Check if the `videos/` directory exists
   - Ensure you have write permissions
   - Try the test script first

2. **Video file is empty**:
   - Check if your camera is working
   - Try a different codec if available
   - Restart the application

3. **Poor video quality**:
   - The video uses full quality, no compression
   - Large file sizes are normal
   - Consider post-processing for compression if needed

### System Requirements

- OpenCV with video writing support
- Sufficient disk space for video files
- Camera access permissions

## Integration

The video recording functionality is fully integrated with the existing gaze tracking system:

- **Real-time Processing**: All gaze detection and hand tracking work during recording
- **Visual Feedback**: Recording status is displayed on screen
- **Automatic Cleanup**: Recording stops when the application exits
- **Session Management**: Each recording session is timestamped and saved separately

## Future Enhancements

Potential improvements for video recording:

- **Compression Options**: Add different quality settings
- **Audio Recording**: Include microphone audio
- **Streaming**: Real-time streaming to external services
- **Metadata**: Include gaze tracking data in video metadata
- **Batch Processing**: Process multiple video files 