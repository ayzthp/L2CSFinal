#!/usr/bin/env python3
"""
Simple Gaze-Hand Focus Tracker
Based on the working realtime_gaze_tracker.py with hand detection added
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
import time
import math
from typing import Optional, Tuple

# Import our L2CS components
from l2cs.model import L2CS
from utils.face_detection import FaceDetector
from utils.gaze_estimation import convert_raw_gaze_predictions_to_angles
from utils.transforms import preprocess_face_for_gaze_estimation
from utils.model_utils import load_l2cs_model, download_l2cs_pretrained_model
import os

class SimpleGazeHandTracker:
    def __init__(self, save_dir="captures"):
        """Initialize the simple gaze-hand tracker"""
        print("🚀 Initializing Simple Gaze-Hand Tracker...")
        
        # Create save directory
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"📁 Save directory: {self.save_dir}")
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load L2CS model
        model_path = "pretrained_models/L2CSNet_gaze360.pkl"
        if not os.path.exists(model_path):
            model_path = download_l2cs_pretrained_model()
        
        self.gaze_model = load_l2cs_model(model_path, self.device)
        
        # Initialize face detector
        self.face_detector = FaceDetector()
        
        # Model configuration
        self.model_input_size = (448, 448)
        self.dataset_mean = (0.485, 0.456, 0.406)
        self.dataset_std = (0.229, 0.224, 0.225)
        
        # Initialize MediaPipe for hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Focus detection parameters
        self.focus_threshold = 5
        self.focus_counter = 0
        self.similarity_threshold = 0.85
        self.min_gaze_strength = 0.13  # Minimum gaze vector magnitude for valid detection
        
        print("✅ Simple Gaze-Hand Tracker initialized!")
    
    def detect_right_hand(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect right hand and return its center position"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Check if it's the right hand
                if handedness.classification[0].label == 'Right':
                    # Get hand center (average of wrist and middle finger MCP)
                    wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                    middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    
                    hand_x = int((wrist.x + middle_mcp.x) / 2 * frame.shape[1])
                    hand_y = int((wrist.y + middle_mcp.y) / 2 * frame.shape[0])
                    
                    return (hand_x, hand_y)
        
        return None
    
    def calculate_focus_similarity(self, gaze_vector: np.ndarray, eye_center: Tuple[int, int], 
                                 hand_position: Tuple[int, int]) -> float:
        """Calculate similarity between gaze direction and eye-to-hand vector"""
        # Vector from eye to hand
        hand_vector = np.array([
            hand_position[0] - eye_center[0],
            hand_position[1] - eye_center[1]
        ])
        
        # Normalize vectors
        gaze_norm = np.linalg.norm(gaze_vector)
        hand_norm = np.linalg.norm(hand_vector)
        
        if gaze_norm == 0 or hand_norm == 0:
            return 0.0
        
        gaze_unit = gaze_vector / gaze_norm
        hand_unit = hand_vector / hand_norm
        
        # Calculate cosine similarity
        similarity = np.dot(gaze_unit, hand_unit)
        return max(0.0, similarity)  # Clamp to positive values
    
    def run(self):
        """Run the gaze-hand tracker"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\\n" + "="*60)
        print("🎯 SIMPLE GAZE-HAND TRACKER STARTED")
        print("="*60)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save frame")
        print("Look at your right hand to activate focus detection!")
        print("="*60)
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Detect faces
                faces = self.face_detector.detect_faces(frame)
                
                # Detect right hand
                right_hand_pos = self.detect_right_hand(frame)
                
                gaze_vector = None
                eye_center = None
                
                # Process first face if available
                if faces:
                    face = faces[0]
                    x1, y1, x2, y2, confidence = face
                    
                    # Draw face bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f"Face: {confidence:.2f}", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Extract and process face
                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size > 0:
                        # Preprocess face image
                        face_tensor = preprocess_face_for_gaze_estimation(
                            face_img, self.model_input_size, 
                            self.dataset_mean, self.dataset_std
                        )
                        
                        # Get gaze prediction
                        with torch.no_grad():
                            face_tensor = face_tensor.to(self.device)
                            yaw_pred, pitch_pred = self.gaze_model(face_tensor)
                            
                            # Convert to angles
                            pitch_angle, yaw_angle = convert_raw_gaze_predictions_to_angles(
                                yaw_pred.cpu().numpy(), 
                                pitch_pred.cpu().numpy()
                            )
                        
                        # Calculate eye center and gaze vector
                        eye_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        
                        # Convert to gaze vector (corrected directions)
                        yaw_rad = math.radians(yaw_angle)
                        pitch_rad = math.radians(pitch_angle)
                        gaze_x = -math.sin(yaw_rad)  # Corrected X
                        gaze_y = -math.sin(pitch_rad)  # Corrected Y
                        gaze_vector = np.array([gaze_x, gaze_y])
                        
                        # Draw gaze arrow (default blue)
                        arrow_length = 150
                        end_x = int(eye_center[0] + arrow_length * gaze_x)
                        end_y = int(eye_center[1] + arrow_length * gaze_y)
                        cv2.arrowedLine(frame, eye_center, (end_x, end_y), (255, 0, 0), 2, tipLength=0.3)
                        
                        # Display gaze angles
                        cv2.putText(frame, f"Gaze: Y={yaw_angle:.1f}° P={pitch_angle:.1f}°", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw right hand if detected
                if right_hand_pos:
                    cv2.circle(frame, right_hand_pos, 10, (0, 255, 0), -1)
                    cv2.putText(frame, "Right Hand", (right_hand_pos[0]-30, right_hand_pos[1]-20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Check for focus if both gaze and hand are detected
                is_focusing = False
                if gaze_vector is not None and eye_center is not None and right_hand_pos is not None:
                    # Check gaze strength (arrow length)
                    gaze_strength = np.linalg.norm(gaze_vector)
                    
                    # Display gaze strength and similarity
                    cv2.putText(frame, f"Gaze Strength: {gaze_strength:.2f}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Only proceed if gaze is strong enough
                    if gaze_strength >= self.min_gaze_strength:
                        similarity = self.calculate_focus_similarity(gaze_vector, eye_center, right_hand_pos)
                        
                        # Display similarity
                        cv2.putText(frame, f"Similarity: {similarity:.2f}", 
                                  (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        if similarity > self.similarity_threshold:
                            self.focus_counter += 1
                        else:
                            self.focus_counter = 0
                    else:
                        # Gaze too weak, reset counter
                        self.focus_counter = 0
                        cv2.putText(frame, "Gaze too weak", 
                                  (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Check if focus is confirmed
                    if self.focus_counter >= self.focus_threshold:
                        is_focusing = True
                        
                        # Draw green gaze arrow
                        arrow_length = 150
                        gaze_x, gaze_y = gaze_vector
                        end_x = int(eye_center[0] + arrow_length * gaze_x)
                        end_y = int(eye_center[1] + arrow_length * gaze_y)
                        cv2.arrowedLine(frame, eye_center, (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)
                        
                        # Draw green lines to hand
                        cv2.line(frame, eye_center, right_hand_pos, (0, 255, 0), 3)
                        
                        # Display focus message
                        cv2.putText(frame, "FOCUSING ON RIGHT HAND!", 
                                  (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # Draw focus counter
                        cv2.putText(frame, f"Focus: {self.focus_counter}/{self.focus_threshold}", 
                                  (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        # Display focus counter when building up
                        cv2.putText(frame, f"Focus Counter: {self.focus_counter}/{self.focus_threshold}", 
                                  (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    self.focus_counter = 0
                
                # Display status
                status_color = (0, 255, 0) if is_focusing else (255, 255, 255)
                status_text = "FOCUSED" if is_focusing else "TRACKING"
                cv2.putText(frame, f"Status: {status_text}", 
                          (frame.shape[1]-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                # Display FPS
                if frame_count % 30 == 0:
                    fps = frame_count / (current_time - start_time)
                    cv2.putText(frame, f"FPS: {fps:.1f}", 
                              (frame.shape[1]-100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Simple Gaze-Hand Tracker', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Create timestamped filename
                    timestamp = int(time.time())
                    status = "focused" if is_focusing else "tracking"
                    filename = f"gaze_hand_{status}_{timestamp}.jpg"
                    filepath = os.path.join(self.save_dir, filename)
                    
                    # Save frame
                    cv2.imwrite(filepath, frame)
                    print(f"📸 Frame saved: {filepath}")
                    
                    # Show save confirmation on frame for next few frames
                    cv2.putText(frame, f"SAVED: {filename}", 
                              (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        except KeyboardInterrupt:
            print("\\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Display summary
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            
            print("\\n" + "="*50)
            print("📊 SESSION SUMMARY")
            print("="*50)
            print(f"  Frames processed: {frame_count}")
            print(f"  Total time: {total_time:.1f} seconds")
            print(f"  Average FPS: {avg_fps:.1f}")
            print("="*50)
            print("🏁 Simple Gaze-Hand Tracker ended.")

def main():
    tracker = SimpleGazeHandTracker()
    tracker.run()

if __name__ == "__main__":
    main() 