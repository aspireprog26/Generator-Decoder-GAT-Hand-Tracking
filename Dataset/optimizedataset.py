import sys
import cv2
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, r"C:\Users\Test\Documents\Hand-Tracking-2\Model")

import keypointdetection as kp  # type: ignore
from constrain import OptimizeHands

ENGINE = r"C:\Users\Test\Documents\RTMPose\model.engine"
class DatasetOptimizer:
    def __init__(self):
        self.pose = kp.MediaPipe()
        self.data_dir = Path(r"C:\Users\Test\Documents\StereoDataset")
        self.types = ["Clean", "Noisy"]
        self.openCalibration()
    
    def openCalibration(self):
        fs = cv2.FileStorage(r"C:\Users\Test\Documents\Hand-Tracking-2\Stereo\stereo.yml", cv2.FILE_STORAGE_READ)
        self.P1 = fs.getNode("P1").mat()
        self.P2 = fs.getNode("P2").mat()
        self.K1 = fs.getNode("K1").mat()
        self.K2 = fs.getNode("K2").mat()
        self.R1 = fs.getNode("R1").mat()
        self.R2 = fs.getNode("R2").mat()
        self.dist1 = fs.getNode("dist1").mat()
        self.dist2 = fs.getNode("dist2").mat()
        fs.release()

    def project3D(self, pts_left, pts_right):
        pts_left = np.asarray(pts_left, dtype = np.float32).reshape(-1,1,2)
        pts_right = np.asarray(pts_right, dtype = np.float32).reshape(-1,1,2)

        pts_left_rect = cv2.undistortPoints(pts_left, self.K1, self.dist1, R = self.R1, P = self.P1)
        pts_right_rect = cv2.undistortPoints(pts_right, self.K2, self.dist2, R = self.R2, P = self.P2)

        # Flatten back to (N, 2)
        pts_left_rect = pts_left_rect.squeeze(1)
        pts_right_rect = pts_right_rect.squeeze(1)

        # Obtain 4D points and scale to 3D
        points4D = cv2.triangulatePoints(self.P1, self.P2, pts_left_rect.T, pts_right_rect.T)
        points3D = (points4D[:3] / points4D[3]).T * 100
        points3D = np.squeeze(points3D)

        return points3D
    
    def optimize(self):
        print("Starting Optimization Process.")
        count = 0
        for img_type in self.types:
            for img in (self.data_dir / type).glob("*.jpg"):
                try:
                    image = cv2.imread(img, cv2.IMREAD_COLOR)
                    if image is None:
                        continue

                    h, w = image.shape[:2]
                    half = w // 2

                    left = image[:, :half]
                    right = image[:, half:]
                        
                    left_kps = self.pose.get_keypoints(left)
                    right_kps = self.pose.get_keypoints(right)
                    if np.any(left_kps) and np.any(right_kps):
                        points3D = self.project3D(left_kps, right_kps)
                        if not np.all(np.isfinite(points3D)):
                            continue

                        points_proj = points3D.copy()
                        hand_optimizer = OptimizeHands(points3D, left, right)
                        optimized_kps = hand_optimizer.optimize()

                        if optimized_kps is not None:
                            points_optim = (optimized_kps[0] + optimized_kps[1]) / 2
                            save_path = self.data_dir / img_type / f"{img.stem}.npy"
                            points = (points_proj, points_optim)
                            np.save(save_path, points)
                        else:
                            continue
                    else:
                        continue
                    count += 1
                except Exception as e:
                    print(f"Error processing {img.name}: {e}")
                    continue
        print("Dataset Optimization Complete.")

    def remove_old(self):
        for type in self.types:
            for img in (self.data_dir / type).glob("*.npy"):
                save_path = self.data_dir / type / f"{img.stem}.npy"
                os.remove(save_path)

optimizer = DatasetOptimizer()
#optimizer.optimize()
#optimizer.remove_old()