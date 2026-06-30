import cv2
import sys
import torch
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy.stats import zscore

sys.path.insert(0, r"/home/mrtcloud-1/Documents/Hand-Tracking-2/Model")
import keypointdetection as kp

BG_COUNT = 6
POSES = ["Counting", "Random"]
DIR = Path("/home/mrtcloud-1/Documents/StereoDataset/")
LABEL_DIR = DIR / "labels"
TENSOR_DIR = DIR / "Tensors"
ENGINE = r"/home/mrtcloud-1/Documents/RTMPoseONNX/rtmpose_hand.trt"

BB_CAM_FX =  822.79041
BB_CAM_FY = BB_CAM_FX
BB_CAM_CX = 318.47345
BB_CAM_CY = 250.31296

ROTATION_VEC_BB = np.array([0.00531, -0.01196, 0.00301], dtype = np.float32)
ROTATION_MAT_BB, _ = cv2.Rodrigues(ROTATION_VEC_BB)
TRANSLATION_VEC_BB = np.array([-24.0381, -0.4563, -1.2326], dtype = np.float32)

RT_CAM_FX_LEFT = 591.6487
RT_CAM_FY_LEFT = 590.5185
RT_CAM_FX_RIGHT = 589.8875
RT_CAM_FY_RIGHT = 588.7449

RT_CAM_CX_LEFT = 383.6427
RT_CAM_CY_LEFT = 306.1255
RT_CAM_CX_RIGHT = 383.7412
RT_CAM_CY_RIGHT = 288.4137
                            
x_coords_left = []
y_coords_left = []
x_coords_right = []
y_coords_right = []

train_mean_x_left = 0
train_mean_y_left = 0
train_mean_x_right = 0
train_mean_y_right = 0

train_std_x_left = 0
train_std_y_left = 0
train_std_x_right = 0
train_std_y_right = 0

targets = []

rtmpose = kp.RTMPose(ENGINE)
tensor_count = 0 

def rigid_transform(orig_coords):
    keypoints_bb = (ROTATION_MAT_BB @ orig_coords.T).T + TRANSLATION_VEC_BB     # Transpose to get (3, 3) x (3, 21) then transpose again for (21, 3) + (3, ) = (21, 3)
    return keypoints_bb

def load_coords(bg, pose, frame):
    dir = LABEL_DIR / f"B{bg}{pose}_BB.mat"
    coords = loadmat(str(dir))["handPara"]
    coords_transposed = np.transpose(coords, (2, 1, 0))     # Turns into shape (1500, 21, 3)
    return coords_transposed[frame, ...]                    # Returns shape (21, 3)

def normalize(x, y):
    x_new = (x - BB_CAM_CX) / BB_CAM_FX
    y_new = (y - BB_CAM_CY) / BB_CAM_FY
    return (x_new, y_new)

def getTargetVector(coords_3d):
    thumb_cmc = coords_3d[1, :]
    thumb_mcp = coords_3d[2, :]
    thumb_ip = coords_3d[3, :]
    thumb_tip = coords_3d[4, :]

    thumb_cmc_unit = thumb_cmc / np.linalg.norm(thumb_cmc)
    thumb_mcp_unit = thumb_mcp / np.linalg.norm(thumb_mcp)
    thumb_ip_unit = thumb_ip / np.linalg.norm(thumb_ip)
    thumb_tip_unit = thumb_tip / np.linalg.norm(thumb_tip)

    

def appendData():
    count = 0
    for bg in range(1, BG_COUNT + 1):
        for pose in POSES:
            for n in range(1500):
                left_img_dir = DIR / f"B{bg}{pose}" / "Left" / f"BB_left_{n}.png"
                right_img_dir = DIR / f"B{bg}{pose}" / "Right" / f"BB_right_{n}.png"
                
                left_img = cv2.imread(str(left_img_dir), cv2.IMREAD_COLOR)
                right_img = cv2.imread(str(right_img_dir), cv2.IMREAD_COLOR)
                kps = rtmpose.get_keypoints(left_img, right_img)

                left_kps = kps[0][0]
                right_kps = kps[0][1]

                for i in range(21):
                    x_left = left_kps[i][0]
                    y_left = left_kps[i][1]
                    x_right = right_kps[i][0]
                    y_right = right_kps[i][1]
                    
                    x_norm_left, y_norm_left = normalize(x_left, y_left)
                    x_norm_right, y_norm_right = normalize(x_right, y_right)

                    x_coords_left.append(x_norm_left)
                    y_coords_left.append(y_norm_left)
                    x_coords_right.append(x_norm_right)
                    y_coords_right.append(y_norm_right)

                coords_3d = load_coords(bg, pose, n)
                target_vector = getTargetVector(coords_3d)
                targets.append(target_vector)
                print(count)
                count += 1


                