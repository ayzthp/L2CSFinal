#!/usr/bin/env python3
"""
Left Hand Focus Bisector Tracker with Bounding Box
Enhanced version using MediaPipe FaceMesh for exact pupil and nose-tip landmarks
Focused on left hand detection with bounding box and posture alignment
Optimized for smooth left-looking detection with reduced delay
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
import time
import math
from typing import Optional, Tuple, List
from collections import deque

# Import our L2CS components
from l2cs.model import L2CS
from utils.face_detection import FaceDetector
from utils.gaze_estimation import convert_raw_gaze_predictions_to_angles
from utils.transforms import preprocess_face_for_gaze_estimation
from utils.model_utils import load_l2cs_model, download_l2cs_pretrained_model
import os

class SmoothGazeFilter:
    """Smoothing filter for gaze vectors to reduce jitter and improve left-looking detection"""
    def __init__(self, window_size=5, alpha=0.7):
        self.window_size = window_size
        self.alpha = alpha  # Exponential smoothing factor
        self.gaze_history = deque(maxlen=window_size)
        self.smoothed_gaze = None
        
    def update(self, gaze_vector: np.ndarray) -> np.ndarray:
        """Update and return smoothed gaze vector"""
        if gaze_vector is None:
            return self.smoothed_gaze
            
        self.gaze_history.append(gaze_vector.copy())
        
        if len(self.gaze_history) < 2:
            self.smoothed_gaze = gaze_vector.copy()
            return self.smoothed_gaze
        
        # Apply exponential smoothing for immediate response
        if self.smoothed_gaze is None:
            self.smoothed_gaze = gaze_vector.copy()
        else:
            self.smoothed_gaze = self.alpha * gaze_vector + (1 - self.alpha) * self.smoothed_gaze
        
        # Apply moving average for stability
        if len(self.gaze_history) >= self.window_size:
            avg_gaze = np.mean(self.gaze_history, axis=0)
            # Blend exponential smoothing with moving average
            self.smoothed_gaze = 0.8 * self.smoothed_gaze + 0.2 * avg_gaze
        
        return self.smoothed_gaze.copy()

class DesktopCamera8GazeTracker:
    def __init__(self, save_dir="captures"):
        """Initialize the desktop camera 8 gaze tracker"""
        print("🚀 Initializing Desktop Camera 8 Gaze Tracker...")
        
        # Camera configuration - FORCE camera 8
        self.camera_id = 8
        print(f"📷 Camera ID: {self.camera_id} (Desktop)")
        
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
        
        # Face detector for cropping
        self.face_detector = FaceDetector()
        
        # Model config
        self.model_input_size = (448, 448)
        self.dataset_mean = (0.485, 0.456, 0.406)
        self.dataset_std = (0.229, 0.224, 0.225)
        
        # MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Focus & alignment params
        self.focus_threshold = 5
        self.focus_counter = 0
        self.similarity_threshold = 0.85
        self.min_gaze_strength = 0.02
        self.alignment_threshold = 10  # degrees
        
        # Smoothing filters for better left-looking detection
        self.gaze_filter = SmoothGazeFilter(window_size=3, alpha=0.8)  # Faster response for left-looking
        self.left_looking_threshold = 0.3  # Threshold for detecting left-looking
        self.left_looking_counter = 0
        self.left_looking_frames = 0
        
        print("✅ Desktop Camera 8 Gaze Tracker initialized!")
    
    def detect_left_hand_with_bbox(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int, int, int]]]:
        """
        Detect left hand and return its center position and bounding box
        Returns: (center_position, bounding_box) where bounding_box is (x, y, w, h)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        
        if res.multi_hand_landmarks and res.multi_handedness:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                if hd.classification[0].label == 'Left':
                    w, h = frame.shape[1], frame.shape[0]
                    
                    # Get all hand landmarks for bounding box calculation
                    landmarks = []
                    for landmark in lm.landmark:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        landmarks.append((x, y))
                    
                    # Calculate bounding box
                    x_coords = [landmark[0] for landmark in landmarks]
                    y_coords = [landmark[1] for landmark in landmarks]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Add padding to bounding box
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    
                    # Calculate center (using wrist and middle finger MCP as before)
                    wrist = lm.landmark[self.mp_hands.HandLandmark.WRIST]
                    mcp = lm.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    center_x = int((wrist.x + mcp.x)/2 * w)
                    center_y = int((wrist.y + mcp.y)/2 * h)
                    
                    return ((center_x, center_y), (x_min, y_min, bbox_width, bbox_height))
        
        return None
    
    def calculate_focus_similarity(self, gaze: np.ndarray, center: Tuple[int,int], hand: Tuple[int,int]) -> float:
        """Cosine similarity between gaze vector and eye_center→hand"""
        hv = np.array([hand[0]-center[0], hand[1]-center[1]])
        gn, hn = np.linalg.norm(gaze), np.linalg.norm(hv)
        if gn==0 or hn==0: return 0.0
        return max(0.0, float(np.dot(gaze/gn, hv/hn)))
    
    def angle_between(self, u: Tuple[float,float], v: Tuple[float,float]) -> Optional[float]:
        """Angle (deg) between 2D vectors"""
        dot = u[0]*v[0] + u[1]*v[1]
        nu, nv = math.hypot(*u), math.hypot(*v)
        if nu*nv==0: return None
        return math.degrees(math.acos(max(-1, min(1, dot/(nu*nv)))))
    
    def calculate_alignment_score(self, LP, RP, NT, hand, gaze: np.ndarray) -> float:
        """Score = |angle(LP→hand, gaze) - angle(RP→hand, gaze)|"""
        L = (hand[0]-LP[0], hand[1]-LP[1])
        R = (hand[0]-RP[0], hand[1]-RP[1])
        V = tuple(gaze)
        t1 = self.angle_between(L, V) or 0
        t2 = self.angle_between(R, V) or 0
        return abs(t1 - t2)
    
    def draw_hand_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                      center: Tuple[int, int], is_focused: bool = False):
        """Draw bounding box around hand with enhanced visualization"""
        x, y, w, h = bbox
        
        # Choose color based on focus state
        if is_focused:
            bbox_color = (0, 255, 0)  # Green when focused
            text_color = (0, 255, 0)
        else:
            bbox_color = (255, 255, 0)  # Yellow when not focused
            text_color = (255, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_color, 2)
        
        # Draw corner indicators
        corner_size = 8
        # Top-left corner
        cv2.line(frame, (x, y), (x + corner_size, y), bbox_color, 3)
        cv2.line(frame, (x, y), (x, y + corner_size), bbox_color, 3)
        # Top-right corner
        cv2.line(frame, (x + w - corner_size, y), (x + w, y), bbox_color, 3)
        cv2.line(frame, (x + w, y), (x + w, y + corner_size), bbox_color, 3)
        # Bottom-left corner
        cv2.line(frame, (x, y + h - corner_size), (x, y + h), bbox_color, 3)
        cv2.line(frame, (x, y + h), (x + corner_size, y + h), bbox_color, 3)
        # Bottom-right corner
        cv2.line(frame, (x + w - corner_size, y + h), (x + w, y + h), bbox_color, 3)
        cv2.line(frame, (x + w, y + h - corner_size), (x + w, y + h), bbox_color, 3)
        
        # Draw center point
        cv2.circle(frame, center, 6, bbox_color, -1)
        cv2.circle(frame, center, 8, bbox_color, 2)
        
        # Add hand label with background
        label = "LEFT HAND"
        if is_focused:
            label += " - FOCUSED"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw text background
        text_bg_x1 = x
        text_bg_y1 = y - text_height - 10
        text_bg_x2 = x + text_width + 10
        text_bg_y2 = y
        
        cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), bbox_color, -1)
        cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), bbox_color, 2)
        
        # Draw text
        cv2.putText(frame, label, (x + 5, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add bounding box dimensions
        dim_text = f"{w}×{h}"
        cv2.putText(frame, dim_text, (x + w + 5, y + h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    def run(self):
        # FORCE camera 8 for desktop
        print(f"🎯 Attempting to use Camera {self.camera_id} for desktop...")
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print(f"❌ Camera {self.camera_id} is not available!")
            print("🔧 Please ensure camera 8 is connected and available.")
            print("💡 You may need to:")
            print("   - Connect an external camera")
            print("   - Check camera permissions")
            print("   - Restart the application")
            return
        
        print(f"✅ Successfully connected to Camera {self.camera_id}")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS for smoother detection
        print("🎯 STARTED — look at your left hand")
        print("📦 Bounding box will be drawn around detected left hand")
        print("👁️ Optimized for smooth left-looking detection")
        print(f"📷 Using camera {self.camera_id} (Desktop)")
        
        cnt = 0; t0 = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                cnt += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect left hand with bounding box
                hand_data = self.detect_left_hand_with_bbox(frame)
                lh = None
                bbox = None
                
                if hand_data:
                    lh, bbox = hand_data
                
                # FaceMesh
                mesh = self.face_mesh.process(rgb)
                gaze_vec = None; center=None; LP=None; RP=None; NT=None
                if mesh.multi_face_landmarks:
                    m = mesh.multi_face_landmarks[0]
                    h, w = frame.shape[:2]
                    lp_lm, rp_lm, nt_lm = m.landmark[468], m.landmark[473], m.landmark[1]
                    LP = (int(lp_lm.x*w), int(lp_lm.y*h))
                    RP = (int(rp_lm.x*w), int(rp_lm.y*h))
                    NT = (int(nt_lm.x*w), int(nt_lm.y*h))
                    center = ((LP[0]+RP[0])//2, (LP[1]+RP[1])//2)
                    # Draw pupils & nose
                    cv2.circle(frame, LP, 3, (255,0,0), -1)
                    cv2.circle(frame, RP, 3, (255,0,0), -1)
                    cv2.circle(frame, NT, 5, (0,0,255), -1)
                    cv2.circle(frame, center,4,(0,255,255),-1)
                    # Crop face for gaze
                    faces = self.face_detector.detect_faces(frame)
                    if faces:
                        x1,y1,x2,y2,_ = faces[0]
                        face_img = frame[y1:y2, x1:x2]
                        if face_img.size>0:
                            ft = preprocess_face_for_gaze_estimation(face_img,
                                   self.model_input_size, self.dataset_mean, self.dataset_std)
                            with torch.no_grad():
                                ft = ft.to(self.device)
                                yp, pp = self.gaze_model(ft)
                                p_ang, y_ang = convert_raw_gaze_predictions_to_angles(
                                    yp.cpu().numpy(), pp.cpu().numpy())
                            gx = -math.sin(math.radians(y_ang))
                            gy = -math.sin(math.radians(p_ang))
                            raw_gaze_vec = np.array([gx, gy])
                            
                            # Apply smoothing filter for better left-looking detection
                            gaze_vec = self.gaze_filter.update(raw_gaze_vec)
                            
                            # Enhanced left-looking detection
                            is_looking_left = gaze_vec[0] < -self.left_looking_threshold
                            if is_looking_left:
                                self.left_looking_counter += 1
                                self.left_looking_frames += 1
                            else:
                                self.left_looking_counter = max(0, self.left_looking_counter - 1)
                            
                            # draw smoothed blue arrow from nose tip
                            L = 150
                            nx, ny = int(NT[0]+L*gaze_vec[0]), int(NT[1]+L*gaze_vec[1])
                            arrow_color = (0, 255, 255) if is_looking_left else (255, 0, 0)  # Yellow for left-looking
                            cv2.arrowedLine(frame, NT, (nx,ny), arrow_color, 3, tipLength=0.3)
                            
                            # Display gaze angles and left-looking status
                            cv2.putText(frame, f"Gaze: Y={y_ang:.1f}° P={p_ang:.1f}°",
                                        (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                            
                            # Left-looking indicator
                            if is_looking_left:
                                cv2.putText(frame, "LOOKING LEFT", (10, 55),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                cv2.putText(frame, f"Left Counter: {self.left_looking_counter}",
                                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # draw left hand
                if lh:
                    self.draw_hand_bbox(frame, bbox, lh, self.focus_counter>=self.focus_threshold)
                
                # focus logic with enhanced left-looking detection
                if gaze_vec is not None and center and lh:
                    strength = np.linalg.norm(gaze_vec)
                    cv2.putText(frame, f"Strength: {strength:.2f}", (10,105),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                    if strength>=self.min_gaze_strength:
                        sim = self.calculate_focus_similarity(gaze_vec, center, lh)
                        cv2.putText(frame, f"Sim: {sim:.2f}", (10,125),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                        
                        # Enhanced focus detection with left-looking bonus
                        focus_bonus = 0.1 if is_looking_left else 0.0  # Bonus for left-looking
                        adjusted_sim = sim + focus_bonus
                        
                        if adjusted_sim > self.similarity_threshold:
                            self.focus_counter += 1
                        else:
                            self.focus_counter = max(0, self.focus_counter - 1)  # Gradual decrease
                    else:
                        self.focus_counter = max(0, self.focus_counter - 1)
                        cv2.putText(frame, "Gaze too weak",(10,125),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                    
                    if self.focus_counter>=self.focus_threshold:
                        # alignment score
                        score = self.calculate_alignment_score(LP, RP, NT, lh, gaze_vec)
                        aligned = score<=self.alignment_threshold
                        col = (0,255,0) if aligned else (255,0,0)
                        # pupils->hand
                        cv2.line(frame, LP, lh, col,2)
                        cv2.line(frame, RP, lh, col,2)
                        # nose->hand arrow
                        ex, ey = int(NT[0]+150*gaze_vec[0]), int(NT[1]+150*gaze_vec[1])
                        cv2.arrowedLine(frame, NT, (ex,ey), col,4,tipLength=0.5)
                        # annotations
                        cv2.putText(frame, f"Score: {score:.1f}°",(10,220),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
                        status = "Perfect Posture" if aligned else "Misaligned Posture"
                        cv2.putText(frame, status,(10,250),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)
                        cv2.putText(frame, "FOCUSING ON LEFT HAND!",(10,280),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
                        cv2.putText(frame, f"Focus: {self.focus_counter}/{self.focus_threshold}",
                                    (10,310),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                        
                        # Left-looking bonus indicator
                        if is_looking_left:
                            cv2.putText(frame, "LEFT-LOOKING BONUS!", (10, 340),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.putText(frame, f"Count: {self.focus_counter}/{self.focus_threshold}",
                                    (10,150),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                else:
                    self.focus_counter = max(0, self.focus_counter - 1)
                
                # status & fps
                st_col = (0,255,0) if self.focus_counter>=self.focus_threshold else (255,255,255)
                st_txt = "FOCUSED" if self.focus_counter>=self.focus_threshold else "TRACKING"
                cv2.putText(frame, f"Status: {st_txt}", (frame.shape[1]-200,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,st_col,2)
                if cnt%30==0:
                    fps = cnt/(time.time()-t0)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-100,60),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                
                cv2.imshow('Desktop Camera 8 Gaze Tracker', frame)
                k = cv2.waitKey(1)&0xFF
                if k==ord('q'): break
                if k==ord('s'):
                    ts = int(time.time())
                    st = "focused" if self.focus_counter>=self.focus_threshold else "tracking"
                    al = "aligned" if (score<=self.alignment_threshold) else "misaligned"
                    left = "left_looking" if is_looking_left else "center"
                    fn = f"desktop_camera8_{st}_{al}_{left}_{ts}.jpg"
                    fp = os.path.join(self.save_dir, fn)
                    cv2.imwrite(fp, frame)
                    print(f"📸 Saved: {fp}")
                    cv2.putText(frame, f"SAVED: {fn}", (10,frame.shape[0]-30),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()
            total = time.time()-t0
            print(f"\n📊 Processed {cnt} frames in {total:.1f}s → {cnt/total:.1f} FPS")
            print(f"👁️ Left-looking frames: {self.left_looking_frames}")
            print("🏁 Desktop Camera 8 Tracker ended.")

def main():
    # Desktop Camera 8 Gaze Tracker
    DesktopCamera8GazeTracker().run()

if __name__ == "__main__":
    main()
