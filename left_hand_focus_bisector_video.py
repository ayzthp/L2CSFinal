#!/usr/bin/env python3
"""
Left Hand Focus Bisector Tracker with Video Recording
Enhanced version using MediaPipe FaceMesh for exact pupil and nose-tip landmarks
Focused on left hand detection and posture alignment with video recording capability
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
import time
import math
import os
from typing import Optional, Tuple

# Import our L2CS components
from l2cs.model import L2CS
from utils.face_detection import FaceDetector
from utils.gaze_estimation import convert_raw_gaze_predictions_to_angles
from utils.transforms import preprocess_face_for_gaze_estimation
from utils.model_utils import load_l2cs_model, download_l2cs_pretrained_model

class LeftHandFocusBisectorVideo:
    def __init__(self, save_dir="captures", video_dir="videos"):
        """Initialize the left hand focus bisector tracker with video recording"""
        print("🚀 Initializing Left Hand Focus Bisector Tracker with Video Recording...")
        
        # Create save directories
        self.save_dir = save_dir
        self.video_dir = video_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        print(f"📁 Save directory: {self.save_dir}")
        print(f"📁 Video directory: {self.video_dir}")
        
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
        self.alignment_threshold = 10  # degrees (reduced from 15° for more precision)
        
        # Video recording parameters
        self.is_recording = False
        self.video_writer = None
        self.video_filename = None
        self.recording_start_time = None
        self.recording_frame_count = 0
        self.actual_fps = 15.0  # Default FPS, will be updated dynamically
        
        print("✅ Left Hand Focus Bisector Tracker with Video Recording initialized!")
    
    def detect_left_hand(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect left hand and return its center position"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if res.multi_hand_landmarks and res.multi_handedness:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                if hd.classification[0].label == 'Left':
                    w, h = frame.shape[1], frame.shape[0]
                    wrist = lm.landmark[self.mp_hands.HandLandmark.WRIST]
                    mcp = lm.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    x = int((wrist.x + mcp.x)/2 * w)
                    y = int((wrist.y + mcp.y)/2 * h)
                    return (x, y)
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
    
    def run(self):
        """Run the left hand focus bisector tracker with video recording"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Get actual frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("\n" + "="*60)
        print("🎯 LEFT HAND FOCUS BISECTOR TRACKER WITH VIDEO RECORDING")
        print("="*60)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save frame")
        print("  'r' - Start/Stop video recording")
        print("  'v' - View recording status")
        print("Look at your left hand to activate focus detection!")
        print("Using MediaPipe FaceMesh for exact pupil & nose-tip landmarks")
        print("="*60)
        
        cnt = 0; t0 = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                cnt += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect left hand
                lh = self.detect_left_hand(frame)
                
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
                            gaze_vec = np.array([gx, gy])
                            # draw default blue arrow from nose tip
                            L = 150
                            nx, ny = int(NT[0]+L*gx), int(NT[1]+L*gy)
                            cv2.arrowedLine(frame, NT, (nx,ny), (255,0,0), 2, tipLength=0.3)
                            cv2.putText(frame, f"Gaze: Y={y_ang:.1f}° P={p_ang:.1f}°",
                                        (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                
                # draw left hand
                if lh:
                    cv2.circle(frame, lh, 10, (0,255,0), -1)
                    cv2.putText(frame, "Left Hand", (lh[0]-30, lh[1]-20),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                
                # focus logic
                if gaze_vec is not None and center and lh and LP and RP and NT:
                    strength = np.linalg.norm(gaze_vec)
                    cv2.putText(frame, f"Strength: {strength:.2f}", (10,60),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                    if strength>=self.min_gaze_strength:
                        sim = self.calculate_focus_similarity(gaze_vec, center, lh)
                        cv2.putText(frame, f"Sim: {sim:.2f}", (10,80),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                        if sim>self.similarity_threshold:
                            self.focus_counter+=1
                        else:
                            self.focus_counter=0
                    else:
                        self.focus_counter=0
                        cv2.putText(frame, "Gaze too weak",(10,80),
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
                    else:
                        cv2.putText(frame, f"Count: {self.focus_counter}/{self.focus_threshold}",
                                    (10,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                else:
                    self.focus_counter=0
                
                # status & fps
                st_col = (0,255,0) if self.focus_counter>=self.focus_threshold else (255,255,255)
                st_txt = "FOCUSED" if self.focus_counter>=self.focus_threshold else "TRACKING"
                cv2.putText(frame, f"Status: {st_txt}", (frame.shape[1]-200,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,st_col,2)
                if cnt%30==0:
                    fps = cnt/(time.time()-t0)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-100,60),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                
                # Display recording status
                if self.is_recording:
                    cv2.putText(frame, "RECORDING", (frame.shape[1]-150, frame.shape[0]-30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f"FPS: {self.actual_fps:.1f}", (frame.shape[1]-150, frame.shape[0]-60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.circle(frame, (frame.shape[1]-20, frame.shape[0]-20), 8, (0, 0, 255), -1)
                
                # Write frame to video if recording
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(frame)
                    self.recording_frame_count += 1
                    
                    # Update actual FPS every 30 frames
                    if self.recording_frame_count % 30 == 0:
                        elapsed_time = time.time() - self.recording_start_time
                        if elapsed_time > 0:
                            self.actual_fps = self.recording_frame_count / elapsed_time
                            # Clamp FPS to reasonable range (5-30 FPS)
                            self.actual_fps = max(5.0, min(30.0, self.actual_fps))
                
                cv2.imshow('Left Hand Focus Bisector Tracker with Video Recording', frame)
                k = cv2.waitKey(1)&0xFF
                if k==ord('q'): break
                if k==ord('s'):
                    ts = int(time.time())
                    st = "focused" if self.focus_counter>=self.focus_threshold else "tracking"
                    al = "aligned" if (score<=self.alignment_threshold) else "misaligned"
                    fn = f"left_bisector_{st}_{al}_{ts}.jpg"
                    fp = os.path.join(self.save_dir, fn)
                    cv2.imwrite(fp, frame)
                    print(f"📸 Saved: {fp}")
                    cv2.putText(frame, f"SAVED: {fn}", (10,frame.shape[0]-30),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                elif k==ord('r'):
                    # Toggle recording
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording(frame_width, frame_height)
                elif k==ord('v'):
                    # Show recording status
                    if self.is_recording:
                        print(f"🎥 Currently recording: {self.video_filename}")
                    else:
                        print("🎥 Not recording")
        except KeyboardInterrupt:
            pass
        finally:
            # Stop recording if active
            if self.is_recording:
                self.stop_recording()
            
            cap.release()
            cv2.destroyAllWindows()
            total = time.time()-t0
            print(f"\n📊 Processed {cnt} frames in {total:.1f}s → {cnt/total:.1f} FPS")
            print("🏁 Left Hand Focus Bisector Tracker with Video Recording ended.")

    def start_recording(self, frame_width: int, frame_height: int):
        """Start video recording"""
        if not self.is_recording:
            timestamp = int(time.time())
            self.video_filename = f"left_hand_bisector_session_{timestamp}.mp4"
            video_path = os.path.join(self.video_dir, self.video_filename)
            
            # Initialize video writer with H.264 codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Use 15 FPS instead of 30 to match typical processing speed
            self.video_writer = cv2.VideoWriter(video_path, fourcc, 15.0, (frame_width, frame_height))
            self.is_recording = True
            self.recording_start_time = time.time()
            self.recording_frame_count = 0
            print(f"🎥 Started recording: {self.video_filename}")
            print(f"📹 Recording at 15 FPS for normal playback speed")
    
    def stop_recording(self):
        """Stop video recording"""
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            self.is_recording = False
            print(f"🎥 Stopped recording: {self.video_filename}")
            self.video_writer = None
            self.video_filename = None

def main():
    LeftHandFocusBisectorVideo().run()

if __name__ == "__main__":
    main()
