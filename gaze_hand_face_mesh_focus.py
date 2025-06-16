#!/usr/bin/env python3
"""
Gaze-Hand Face Mesh Focus Tracker
Enhanced version using MediaPipe FaceMesh for exact pupil and nose-tip landmarks
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

class GazeHandFaceMeshFocus:
    def __init__(self, save_dir="captures"):
        """Initialize the gaze-hand face mesh focus tracker"""
        print("🚀 Initializing Gaze-Hand Face Mesh Focus Tracker...")
        
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
        
        # Initialize MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Focus detection parameters
        self.focus_threshold = 5
        self.focus_counter = 0
        self.similarity_threshold = 0.85
        self.min_gaze_strength = 0.13  # Minimum gaze vector magnitude for valid detection
        
        # Posture alignment parameters
        self.alignment_threshold = 15  # degrees
        
        print("✅ Gaze-Hand Face Mesh Focus Tracker initialized!")
    
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
    
    def angle_between(self, u: Tuple[float, float], v: Tuple[float, float]) -> Optional[float]:
        """Calculate angle between two 2D vectors in degrees"""
        # u, v: 2D vectors
        dot = u[0] * v[0] + u[1] * v[1]
        nu, nv = math.hypot(*u), math.hypot(*v)
        if nu * nv == 0:
            return None
        cos_theta = max(-1, min(1, dot / (nu * nv)))
        return math.degrees(math.acos(cos_theta))
    
    def calculate_alignment_score(self, left_pupil: Tuple[int, int], right_pupil: Tuple[int, int], 
                                nose_tip: Tuple[int, int], hand_position: Tuple[int, int]) -> float:
        """Calculate posture alignment score based on pupil-nose-hand geometry"""
        # Compute three vectors
        L = (hand_position[0] - left_pupil[0], hand_position[1] - left_pupil[1])
        R = (hand_position[0] - right_pupil[0], hand_position[1] - right_pupil[1])
        C = (hand_position[0] - nose_tip[0], hand_position[1] - nose_tip[1])
        
        # Calculate angles
        theta1 = self.angle_between(L, C)
        theta2 = self.angle_between(R, C)
        
        if theta1 is None or theta2 is None:
            return 180.0  # Maximum misalignment if vectors are invalid
        
        # Score is how far the nose-hand bisector deviates
        score = abs(theta1 - theta2)
        return score
    
    def run(self):
        """Run the gaze-hand tracker"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n" + "="*60)
        print("🎯 GAZE-HAND FACE MESH FOCUS TRACKER STARTED")
        print("="*60)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save frame")
        print("Look at your right hand to activate focus detection!")
        print("Using MediaPipe FaceMesh for exact pupil & nose-tip landmarks")
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
                
                # Convert frame for MediaPipe processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect right hand
                right_hand_pos = self.detect_right_hand(frame)
                
                # Initialize landmark variables
                gaze_vector = None
                eye_center = None
                left_pupil = None
                right_pupil = None
                nose_tip = None
                
                # Process FaceMesh for exact landmarks
                mesh_results = self.face_mesh.process(rgb_frame)
                
                if mesh_results.multi_face_landmarks:
                    mesh = mesh_results.multi_face_landmarks[0]
                    h, w, _ = frame.shape
                    
                    # Extract exact pupil & nose-tip coordinates
                    left_pupil_landmark = mesh.landmark[468]
                    right_pupil_landmark = mesh.landmark[473]
                    nose_tip_landmark = mesh.landmark[1]
                    
                    # Convert to pixel coordinates
                    left_pupil = (int(left_pupil_landmark.x * w), int(left_pupil_landmark.y * h))
                    right_pupil = (int(right_pupil_landmark.x * w), int(right_pupil_landmark.y * h))
                    nose_tip = (int(nose_tip_landmark.x * w), int(nose_tip_landmark.y * h))
                    
                    # Calculate eye center as midpoint between pupils
                    eye_center = ((left_pupil[0] + right_pupil[0]) // 2, (left_pupil[1] + right_pupil[1]) // 2)
                    
                    # Draw landmarks
                    cv2.circle(frame, left_pupil, 3, (255, 0, 0), -1)  # Blue for left pupil
                    cv2.circle(frame, right_pupil, 3, (255, 0, 0), -1)  # Blue for right pupil
                    cv2.circle(frame, nose_tip, 5, (0, 0, 255), -1)  # Red for nose tip
                    cv2.circle(frame, eye_center, 4, (0, 255, 255), -1)  # Yellow for eye center
                    
                    # Get face bounding box for gaze estimation
                    faces = self.face_detector.detect_faces(frame)
                    if faces:
                        face = faces[0]
                        x1, y1, x2, y2, confidence = face
                        
                        # Draw face bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame, f"Face: {confidence:.2f}", (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # Extract and process face for gaze estimation
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
                alignment_score = 0.0
                is_aligned = False
                
                if gaze_vector is not None and eye_center is not None and right_hand_pos is not None and left_pupil is not None and right_pupil is not None and nose_tip is not None:
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
                        
                        # Calculate alignment score using FaceMesh landmarks
                        alignment_score = self.calculate_alignment_score(left_pupil, right_pupil, nose_tip, right_hand_pos)
                        is_aligned = alignment_score <= self.alignment_threshold
                        
                        # Choose colors based on alignment
                        line_color = (0, 255, 0) if is_aligned else (255, 0, 0)
                        text_color = (0, 255, 0) if is_aligned else (255, 0, 0)
                        status_text = "Perfect Posture ✅" if is_aligned else "Posture Misaligned ❌"
                        
                        # Draw lines from both pupils to hand
                        cv2.line(frame, left_pupil, right_hand_pos, line_color, 2)
                        cv2.line(frame, right_pupil, right_hand_pos, line_color, 2)
                        
                        # Draw bisector from nose to hand
                        cv2.line(frame, nose_tip, right_hand_pos, line_color, 2)
                        
                        # Draw gaze vector from nose tip
                        nose_arrow_length = 120  # Slightly shorter than eye center vector
                        nose_end_x = int(nose_tip[0] + nose_arrow_length * gaze_x)
                        nose_end_y = int(nose_tip[1] + nose_arrow_length * gaze_y)
                        cv2.arrowedLine(frame, nose_tip, (nose_end_x, nose_end_y), line_color, 2, tipLength=0.3)
                        
                        # Draw gaze vector from eye center
                        arrow_length = 150
                        gaze_x, gaze_y = gaze_vector
                        end_x = int(eye_center[0] + arrow_length * gaze_x)
                        end_y = int(eye_center[1] + arrow_length * gaze_y)
                        cv2.arrowedLine(frame, eye_center, (end_x, end_y), line_color, 2, tipLength=0.3)
                        
                        # Annotate alignment score and status
                        cv2.putText(frame, f"Alignment Δ: {alignment_score:.1f}°", 
                                  (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                        cv2.putText(frame, status_text, 
                                  (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                        
                        # Display focus message
                        cv2.putText(frame, "FOCUSING ON RIGHT HAND!", 
                                  (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # Draw focus counter
                        cv2.putText(frame, f"Focus: {self.focus_counter}/{self.focus_threshold}", 
                                  (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
                cv2.imshow('Gaze-Hand Face Mesh Focus Tracker', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Create timestamped filename
                    timestamp = int(time.time())
                    status = "focused" if is_focusing else "tracking"
                    aligned = "aligned" if is_aligned else "misaligned"
                    filename = f"gaze_hand_face_mesh_{status}_{aligned}_{timestamp}.jpg"
                    filepath = os.path.join(self.save_dir, filename)
                    
                    # Save frame
                    cv2.imwrite(filepath, frame)
                    print(f"📸 Frame saved: {filepath}")
                    
                    # Show save confirmation on frame for next few frames
                    cv2.putText(frame, f"SAVED: {filename}", 
                              (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Display summary
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            
            print("\n" + "="*50)
            print("📊 SESSION SUMMARY")
            print("="*50)
            print(f"  Frames processed: {frame_count}")
            print(f"  Total time: {total_time:.1f} seconds")
            print(f"  Average FPS: {avg_fps:.1f}")
            print("="*50)
            print("🏁 Gaze-Hand Face Mesh Focus Tracker ended.")

def main():
    tracker = GazeHandFaceMeshFocus()
    tracker.run()

if __name__ == "__main__":
    main() 