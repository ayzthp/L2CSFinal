#!/usr/bin/env python3
"""
Left Hand Focus Bisector Tracker — Saves to MP4
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
import time
import math
import os
from typing import Optional, Tuple

from l2cs.model import L2CS
from utils.face_detection import FaceDetector
from utils.gaze_estimation import convert_raw_gaze_predictions_to_angles
from utils.transforms import preprocess_face_for_gaze_estimation
from utils.model_utils import load_l2cs_model, download_l2cs_pretrained_model

class LeftHandFocusBisector:
    def __init__(self, save_dir="captures"):
        print("🚀 Initializing...")
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📁 Save directory: {self.save_dir} | Using device: {self.device}")

        # Load model
        model_path = "pretrained_models/L2CSNet_gaze360.pkl"
        if not os.path.exists(model_path):
            model_path = download_l2cs_pretrained_model()
        self.gaze_model = load_l2cs_model(model_path, self.device)

        self.face_detector = FaceDetector()
        self.model_input_size = (448, 448)
        self.dataset_mean = (0.485, 0.456, 0.406)
        self.dataset_std = (0.229, 0.224, 0.225)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.focus_threshold = 5
        self.focus_counter = 0
        self.similarity_threshold = 0.85
        self.min_gaze_strength = 0.13
        self.alignment_threshold = 10

    def detect_left_hand(self, frame):
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

    def calculate_focus_similarity(self, gaze, center, hand):
        hv = np.array([hand[0]-center[0], hand[1]-center[1]])
        gn, hn = np.linalg.norm(gaze), np.linalg.norm(hv)
        if gn==0 or hn==0: return 0.0
        return max(0.0, float(np.dot(gaze/gn, hv/hn)))

    def angle_between(self, u, v):
        dot = u[0]*v[0] + u[1]*v[1]
        nu, nv = math.hypot(*u), math.hypot(*v)
        if nu*nv==0: return None
        return math.degrees(math.acos(max(-1, min(1, dot/(nu*nv)))))

    def calculate_alignment_score(self, LP, RP, NT, hand, gaze):
        L = (hand[0]-LP[0], hand[1]-LP[1])
        R = (hand[0]-RP[0], hand[1]-RP[1])
        V = tuple(gaze)
        t1 = self.angle_between(L, V) or 0
        t2 = self.angle_between(R, V) or 0
        return abs(t1 - t2)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 640); cap.set(4, 480)
        print("🎯 STARTED — look at your left hand")

        # 🔴 Video Writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(self.save_dir, 'output_video.mp4')
        fps = 20
        frame_size = (int(cap.get(3)), int(cap.get(4)))
        out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        print(f"📽️ Recording to {video_path}")

        cnt = 0; t0 = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                cnt += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                lh = self.detect_left_hand(frame)
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
                    cv2.circle(frame, LP, 3, (255,0,0), -1)
                    cv2.circle(frame, RP, 3, (255,0,0), -1)
                    cv2.circle(frame, NT, 5, (0,0,255), -1)
                    cv2.circle(frame, center,4,(0,255,255),-1)

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
                            nx, ny = int(NT[0]+150*gx), int(NT[1]+150*gy)
                            cv2.arrowedLine(frame, NT, (nx,ny), (255,0,0), 2, tipLength=0.3)
                            cv2.putText(frame, f"Gaze: Y={y_ang:.1f}° P={p_ang:.1f}°",
                                        (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

                if lh:
                    cv2.circle(frame, lh, 10, (0,255,0), -1)
                    cv2.putText(frame, "Left Hand", (lh[0]-30, lh[1]-20),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

                if gaze_vec is not None and center and lh and LP and RP and NT:
                    strength = np.linalg.norm(gaze_vec)
                    cv2.putText(frame, f"Strength: {strength:.2f}", (10,60),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                    if strength >= self.min_gaze_strength:
                        sim = self.calculate_focus_similarity(gaze_vec, center, lh)
                        cv2.putText(frame, f"Sim: {sim:.2f}", (10,80),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                        self.focus_counter += 1 if sim > self.similarity_threshold else -self.focus_counter
                    else:
                        self.focus_counter = 0
                        cv2.putText(frame, "Gaze too weak", (10,80),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)

                    if self.focus_counter >= self.focus_threshold:
                        score = self.calculate_alignment_score(LP, RP, NT, lh, gaze_vec)
                        aligned = score <= self.alignment_threshold
                        col = (0,255,0) if aligned else (255,0,0)
                        cv2.line(frame, LP, lh, col, 2)
                        cv2.line(frame, RP, lh, col, 2)
                        ex, ey = int(NT[0]+150*gaze_vec[0]), int(NT[1]+150*gaze_vec[1])
                        cv2.arrowedLine(frame, NT, (ex,ey), col, 4, tipLength=0.5)
                        cv2.putText(frame, f"Score: {score:.1f}°", (10,220),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
                        cv2.putText(frame, "FOCUSING ON LEFT HAND!", (10,250),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
                    else:
                        cv2.putText(frame, f"Count: {self.focus_counter}/{self.focus_threshold}",
                                    (10,100), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                else:
                    self.focus_counter = 0

                cv2.putText(frame, f"Status: {'FOCUSED' if self.focus_counter >= self.focus_threshold else 'TRACKING'}",
                            (frame.shape[1]-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,255,0) if self.focus_counter >= self.focus_threshold else (255,255,255), 2)

                if cnt % 30 == 0:
                    fps_now = cnt / (time.time()-t0)
                    cv2.putText(frame, f"FPS: {fps_now:.1f}", (frame.shape[1]-100, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                # 🔴 Save frame to video
                out.write(frame)

                cv2.imshow("Left Hand Focus Bisector Tracker", frame)
                if cv2.waitKey(1)&0xFF == ord('q'):
                    break
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            total = time.time()-t0
            print(f"\n📊 Processed {cnt} frames in {total:.1f}s → {cnt/total:.1f} FPS")
            print(f"🎬 Saved video to {video_path}")
            print("🏁 Tracker ended.")

def main():
    LeftHandFocusBisector().run()

if __name__ == "__main__":
    main()
