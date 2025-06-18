#!/usr/bin/env python3
"""
Simple test script to verify video recording functionality
"""

import cv2
import time
import os

def test_video_recording():
    """Test basic video recording functionality"""
    print("🎥 Testing video recording functionality...")
    
    # Create video directory
    video_dir = "videos"
    os.makedirs(video_dir, exist_ok=True)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📐 Frame dimensions: {frame_width}x{frame_height}")
    
    # Initialize video writer
    timestamp = int(time.time())
    video_filename = f"test_recording_{timestamp}.mp4"
    video_path = os.path.join(video_dir, video_filename)
    
    # Use mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (frame_width, frame_height))
    
    if not video_writer.isOpened():
        print("❌ Could not initialize video writer")
        cap.release()
        return
    
    print(f"🎥 Starting test recording: {video_filename}")
    print("Press 'q' to quit, 'r' to stop recording")
    
    frame_count = 0
    start_time = time.time()
    is_recording = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Add timestamp to frame
            current_time = time.time() - start_time
            cv2.putText(frame, f"Time: {current_time:.1f}s", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show recording status
            if is_recording:
                cv2.putText(frame, "RECORDING", (frame.shape[1]-150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.circle(frame, (frame.shape[1]-20, 20), 8, (0, 0, 255), -1)
            
            # Write frame if recording
            if is_recording:
                video_writer.write(frame)
            
            cv2.imshow('Video Recording Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if is_recording:
                    is_recording = False
                    print("⏹️  Recording stopped")
                else:
                    is_recording = True
                    print("▶️  Recording resumed")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        if is_recording:
            video_writer.release()
            print(f"🎥 Test recording saved: {video_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Summary
        total_time = time.time() - start_time
        fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n📊 Test Summary:")
        print(f"  Frames recorded: {frame_count}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Video file: {video_path}")
        
        # Check if video file exists and has size
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            print(f"  File size: {file_size:,} bytes")
            if file_size > 0:
                print("✅ Video recording test successful!")
            else:
                print("❌ Video file is empty")
        else:
            print("❌ Video file not found")

if __name__ == "__main__":
    test_video_recording() 