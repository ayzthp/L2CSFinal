#!/usr/bin/env python3
"""
Simple OpenCV Test Script
Test OpenCV installation and basic functionality
"""

import cv2
import numpy as np
import sys

def test_opencv_basic():
    """Test basic OpenCV functionality"""
    print("🔍 Testing OpenCV Basic Functionality...")
    
    try:
        # Test version
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        # Test basic image operations
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (10, 10), (90, 90), (0, 255, 0), 2)
        cv2.circle(img, (50, 50), 20, (255, 0, 0), -1)
        
        print("✅ Basic image operations: OK")
        
        # Test window creation (non-blocking)
        cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
        cv2.imshow('Test', img)
        cv2.waitKey(100)  # Show for 100ms
        cv2.destroyAllWindows()
        
        print("✅ Window operations: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_camera_access():
    """Test camera access"""
    print("\n📷 Testing Camera Access...")
    
    # Test camera 8 (desktop)
    cap = cv2.VideoCapture(8)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Camera 8: {width}x{height} @ {fps:.1f} FPS")
        
        # Try to read a frame
        ret, frame = cap.read()
        if ret:
            print("✅ Frame capture: OK")
        else:
            print("⚠️  Frame capture: Failed")
        
        cap.release()
        return True
    else:
        print("❌ Camera 8: Not available")
        
        # Try camera 0 as fallback
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera 0: Available (fallback)")
            cap.release()
            return True
        else:
            print("❌ Camera 0: Not available")
            return False

def test_mediapipe_integration():
    """Test MediaPipe integration with OpenCV"""
    print("\n🤖 Testing MediaPipe Integration...")
    
    try:
        import mediapipe as mp
        
        # Test FaceMesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ FaceMesh: OK")
        
        # Test Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ Hands: OK")
        
        return True
        
    except ImportError:
        print("❌ MediaPipe: Not installed")
        return False
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🎯 OpenCV Installation Test")
    print("="*40)
    
    # Run all tests
    basic_ok = test_opencv_basic()
    camera_ok = test_camera_access()
    mediapipe_ok = test_mediapipe_integration()
    
    print("\n" + "="*40)
    print("📊 Test Results:")
    print(f"  Basic OpenCV: {'✅ OK' if basic_ok else '❌ FAILED'}")
    print(f"  Camera Access: {'✅ OK' if camera_ok else '❌ FAILED'}")
    print(f"  MediaPipe: {'✅ OK' if mediapipe_ok else '❌ FAILED'}")
    
    if basic_ok and camera_ok and mediapipe_ok:
        print("\n🎉 All tests passed! OpenCV is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 