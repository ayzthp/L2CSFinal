#!/usr/bin/env python3
"""
System Check and Monitoring Script for L2CS Gaze Tracking
"""

import cv2
import torch
import mediapipe as mp
import time
import os
import psutil
import platform

def check_system():
    """Check system components"""
    print("🔍 L2CS System Check")
    print("="*50)
    
    # Check Python environment
    print("🐍 Python Environment:")
    print(f"  Python: {platform.python_version()}")
    print(f"  Platform: {platform.platform()}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"  ✅ CUDA: {torch.version.cuda} ({torch.cuda.get_device_name(0)})")
    else:
        print("  ⚠️  CUDA: Not available (using CPU)")
    
    # Check system resources
    print("\n💻 System Resources:")
    print(f"  CPU: {psutil.cpu_count()} cores, {psutil.cpu_percent(interval=1):.1f}% usage")
    mem = psutil.virtual_memory()
    print(f"  Memory: {mem.available//(1024**3)}GB available of {mem.total//(1024**3)}GB")
    disk = psutil.disk_usage('/')
    print(f"  Disk: {disk.free//(1024**3)}GB free of {disk.total//(1024**3)}GB")
    
    # Check camera
    print(f"\n📷 Camera System (ID: 8):")
    cap = cv2.VideoCapture(8)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  ✅ Camera 8: {width}x{height} @ {fps:.1f} FPS")
        cap.release()
    else:
        print("  ❌ Camera 8: Not available")
    
    # Check MediaPipe
    print("\n🤖 MediaPipe Components:")
    try:
        mp_face_mesh = mp.solutions.face_mesh
        print("  ✅ FaceMesh: Available")
    except:
        print("  ❌ FaceMesh: Not available")
    
    try:
        mp_hands = mp.solutions.hands
        print("  ✅ Hands: Available")
    except:
        print("  ❌ Hands: Not available")
    
    # Check L2CS model
    print("\n🧠 L2CS Model:")
    model_path = "pretrained_models/L2CSNet_gaze360.pkl"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  ✅ Model file: {size_mb:.1f} MB")
    else:
        print("  ❌ Model file: Not found")
    
    # Check directories
    print("\n📁 Directories:")
    dirs = ["captures", "videos", "pretrained_models", "l2cs", "utils"]
    for dir_name in dirs:
        if os.path.exists(dir_name):
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/")
    
    print("\n" + "="*50)
    print("✅ System check completed!")

def performance_test(duration=10):
    """Run performance test"""
    print(f"\n⚡ Performance Test ({duration}s)")
    print("="*30)
    
    cap = cv2.VideoCapture(8)
    if not cap.isOpened():
        print("❌ Cannot open camera for performance test")
        return
    
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    start_time = time.time()
    frame_count = 0
    
    print("Starting test... Press 'q' to quit early")
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh = face_mesh.process(rgb)
        
        frame_count += 1
        
        # Show preview
        cv2.putText(frame, f"Test: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Performance Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time
    memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    
    print(f"\n📊 Results:")
    print(f"  FPS: {fps:.1f}")
    print(f"  Frames: {frame_count}")
    print(f"  Memory: {memory_mb:.1f} MB")
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function"""
    print("🎯 L2CS System Check and Monitoring")
    print("="*50)
    
    print("🔧 Options:")
    print("1. System check")
    print("2. Performance test")
    print("3. Both")
    print("4. Exit")
    print()
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        check_system()
    elif choice == "2":
        performance_test()
    elif choice == "3":
        check_system()
        performance_test()
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Running system check...")
        check_system()

if __name__ == "__main__":
    main()
