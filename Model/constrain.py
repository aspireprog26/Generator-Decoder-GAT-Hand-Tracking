import cv2
import numpy as np
import mediapipe as mp 
from optimize import Optimizer

class Constrain:
    def __init__(self, img: np.ndarray, mp_conf: float):
        self.img = img
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(static_image_mode = False, max_num_hands = 1, min_detection_confidence = mp_conf)

    def mp_keypoints(self):
        hand_detected = False
        keypoints = None
        
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_detected = True
            # Obtain the first detected hand keypoints
            hand_kps = results.multi_hand_landmarks[0]
            keypoints = np.zeros((21, 3))               # Create array to store coordinates
            for i, kp in enumerate(hand_kps.landmark):
                keypoints[i] = [kp.x, kp.y, kp.z]

        return hand_detected, keypoints

    def close_mp(self): 
        self.hands.close()
        
class OptimizeHands:
    def __init__(self, stereo_coords: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray):
        w1 = 0.2
        w2 = 0.4
        w3 = 0.3
        w4 = 0.6
        lr = 1e-2
        conf = 0.3
        num_steps = 275

        self.stereo_coords = stereo_coords
        self.left_constraint = Constrain(left_hand, conf)
        self.right_constraint = Constrain(right_hand, conf)
        self.optimizer = Optimizer(w1, w2, w3, w4, lr, num_steps)

    def optimize(self):
        try:
            left_mp = self.left_constraint.mp_keypoints()
            right_mp = self.right_constraint.mp_keypoints()
            if (left_mp[0] == True and right_mp[0] == True) and (left_mp[1] is not None) and (right_mp[1] is not None):                     # Checks if hand is detected by both views
                left_mp_kps = left_mp[1]
                right_mp_kps = right_mp[1]
                avg_mp_kps = (left_mp_kps + right_mp_kps) / 2

                self.left_stereo_optim = self.optimizer.optimize(self.stereo_coords, left_mp_kps)
                self.right_stereo_optim = self.optimizer.optimize(self.stereo_coords, right_mp_kps)

                return [self.left_stereo_optim, self.right_stereo_optim, avg_mp_kps]
            return None
        
        finally:
            self.left_constraint.close_mp()
            self.right_constraint.close_mp()
