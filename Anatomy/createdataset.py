import cv2
import sys
import torch
import numpy as np
from pathlib import Path
from scipy.io import loadmat

sys.path.insert(1, r"C:\Users\Test\Documents\Hand-Tracking\Model")
import keypointdetection as kp

BG_COUNT = 6
POSES = ["Counting", "Random"]
DIR = Path("/home/mrtcloud-1/Documents/StereoDataset/")

rotation_vector = np.array([0.00531, -0.01196, 0.00301], dtype = np.float32)
rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
translation_vector = np.array([-24.0381, -0.4563, -1.2326], dtype = np.float32)

def rigidTransform(orig_coords):
    keypoints_bb = (rotation_matrix @ orig_coords.T).T + translation_vector
    return keypoints_bb

