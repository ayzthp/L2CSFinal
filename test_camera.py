#!/usr/bin/env python3
"""
Camera Test Script
Test available cameras and their properties
"""

import cv2
import time

def test_cameras(max_cameras=10):
    """Test cameras from 0 to max_cameras-1"""
    print("🔍 Testing available cameras...")
    print("=" * 50)
    
    working_cameras = []
    
    for camera_id in range(max_cameras):
        print(f"Testing camera {camera_id}...", end=" ")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Not available")
            continue
        
        # Try to read a frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read frames")
            cap.release()
            continue
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Available - {width}x{height} @ {fps:.1f} FPS")
        working_cameras.append(camera_id)
        
        cap.release()
    
    print("=" * 50)
    print(f"📷 Found {len(working_cameras)} working camera(s): {working_cameras}")
    
    if working_cameras:
        print("\n🎯 To use a specific camera, modify the camera_id parameter:")
        print("   LeftHandFocusBisectorWithBBox(camera_id=YOUR_CAMERA_ID)")
        print("\n📝 Current default: camera_id=8 (for desktop)")
    
    return working_cameras

def preview_camera(camera_id, duration=5):
    """Preview a specific camera for a few seconds"""
    print(f"👁️ Previewing camera {camera_id} for {duration} seconds...")
    print("Press 'q' to quit early")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Camera {camera_id} is not available")
        return
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add camera info to frame
        cv2.putText(frame, f"Camera {camera_id}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(f'Camera {camera_id} Preview', frame)
        
        # Check for quit or timeout
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if time.time() - start_time > duration:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Preview completed")

if __name__ == "__main__":
    print("🎥 Camera Test Utility")
    print("=" * 50)
    
    # Test all cameras
    working_cameras = test_cameras()
    
    if working_cameras:
        print("\n🔧 Options:")
        print("1. Test camera 8 (default for desktop)")
        print("2. Test all working cameras")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            if 8 in working_cameras:
                preview_camera(8)
            else:
                print("❌ Camera 8 is not available")
        elif choice == "2":
            for camera_id in working_cameras:
                preview_camera(camera_id, duration=3)
        else:
            print("👋 Goodbye!")
    else:
        print("❌ No cameras found. Check your camera connections.") 