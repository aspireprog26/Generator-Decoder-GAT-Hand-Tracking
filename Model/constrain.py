import os
import cv2
import time
import platform
import numpy as np
import mediapipe as mp 
from optimize import Optimizer

class Constrain:
    def __init__(self, img, mp_conf: float):
        self.img = img
        self.conf = mp_conf
        self.hand_detected = False
        
        self.mp_keypoints()
    
    def mp_keypoints(self):
        self.mp_hands = mp.solutions.hands
        hands = self.mp_hands.Hands(static_image_mode = False, max_num_hands = 1, min_detection_confidence = self.conf)
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Clear non-harmful MediaPipe warnings from terminal.
        os.system("cls" if platform.system() == "Windows" else "clear")

        if results.multi_hand_landmarks:
            self.hand_detected = True
            # Obtain the first detected hand keypoints
            hand_kps = results.multi_hand_landmarks[0]
            keypoints = np.zeros((21, 3))               # Create array to store coordinates
            for i, kp in enumerate(hand_kps.landmark):
                keypoints[i] = [kp.x, kp.y, kp.z]

        return self.hand_detected, keypoints

    def close_mp(self): 
        self.mp_hands.close()

class OptimizeHands:
    def __init__(self, stereo_coords: np.ndarray, left_hand: str, right_hand: str):
        w1 = 0.35
        w2 = 0.3
        w3 = 0.35
        lr = 1e-2
        conf = 0.2
        num_steps = 100

        self.stereo_coords = stereo_coords
        self.left_stereo_optim = None
        self.right_stereo_optim = None
        self.left_constraint = Constrain(left_hand, conf)
        self.right_constraint = Constrain(right_hand, conf)
        self.optimizer = Optimizer(w1, w2, w3, lr, num_steps)

    def optimize(self):
        left_mp = self.left_constraint.mp_keypoints()
        right_mp = self.right_constraint.mp_keypoints()

        if left_mp[0] and right_mp[0]:                     # Checks if hand is detected by both views
            left_mp_kps = left_mp[1]
            right_mp_kps = right_mp[1]
            avg_mp_kps = (left_mp_kps + right_mp_kps) / 2

            self.left_stereo_optim = self.optimizer.optimize(self.stereo_coords, left_mp_kps)
            self.right_stereo_optim = self.optimizer.optimize(self.stereo_coords, right_mp_kps)

        return [self.left_stereo_optim, self.right_stereo_optim, avg_mp_kps]
    
    def quit(self):
        self.left_constraint.close_mp()
        self.right_constraint.close_mp()