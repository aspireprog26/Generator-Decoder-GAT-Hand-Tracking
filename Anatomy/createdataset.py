import cv2
import sys
import json
import torch
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy.stats import zscore

sys.path.insert(0, r"/home/mrtcloud-1/Documents/Hand-Tracking-2/Model")
import keypointdetection as kp # type: ignore

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

ratio1_means = []
ratio1_stds = []
ratio2_means = []
ratio2_stds = []

targets = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: []
}

target_vecs = []

standardized_stats = {
    "x_train_stand_left": [],
    "y_train_stand_left": [],
    "x_train_stand_right": [],
    "y_train_stand_right": [],
    "ratio1": [],
    "ratio2": []
}

rtmpose = kp.RTMPose(ENGINE)
tensor_count = 0 

def rigidTransform(orig_coords):
    keypoints_bb = (ROTATION_MAT_BB @ orig_coords.T).T + TRANSLATION_VEC_BB     # Transpose to get (3, 3) x (3, 21) then transpose again for (21, 3) + (3, ) = (21, 3)
    return keypoints_bb

def loadCoords(bg, pose, frame):
    dir = LABEL_DIR / f"B{bg}{pose}_BB.mat"
    coords = loadmat(str(dir))["handPara"]
    coords_transposed = np.transpose(coords, (2, 1, 0))     # Turns into shape (1500, 21, 3)
    return coords_transposed[frame, ...]                    # Returns shape (21, 3)

def normalize(x, y):
    x_new = (x - BB_CAM_CX) / BB_CAM_FX
    y_new = (y - BB_CAM_CY) / BB_CAM_FY
    return (x_new, y_new)

def generateTarget(coords_3d):
    for finger in range(5):
        CMC = 1 + 4 * finger
        MCP = 2 + 4 * finger
        IP = 3 + 4 * finger
        TIP = 4 + 4 * finger

        coords_CMC = np.asarray(rigidTransform(coords_3d)[CMC])
        coords_MCP = np.asarray(rigidTransform(coords_3d)[MCP])
        coords_IP = np.asarray(rigidTransform(coords_3d)[IP])
        coords_TIP = np.asarray(rigidTransform(coords_3d)[TIP])
        
        unit_CMC = coords_CMC / np.linalg.norm(coords_CMC)
        unit_MCP = coords_MCP / np.linalg.norm(coords_MCP)
        unit_IP = coords_IP / np.linalg.norm(coords_IP)
        unit_TIP = coords_TIP / np.linalg.norm(coords_TIP)

        cmc_dot_mcp = np.dot(unit_CMC, unit_MCP)
        mcp_dot_ip = np.dot(unit_MCP, unit_IP)
        ip_dot_tip = np.dot(unit_IP, unit_TIP)
        dot_vec = np.array([cmc_dot_mcp, mcp_dot_ip, ip_dot_tip])

        tip_ip_mag = np.linalg.norm(coords_TIP - coords_IP)
        ip_mcp_mag = np.linalg.norm(coords_IP - coords_MCP)
        mcp_cmc_mag = np.linalg.norm(coords_MCP - coords_CMC)

        ratio_1 = tip_ip_mag / ip_mcp_mag
        ratio_2 = ip_mcp_mag / mcp_cmc_mag
        ratio_vec = np.array([ratio_1, ratio_2])

        sgn_tip_ip = np.sign(coords_TIP[2] - coords_IP[2])
        sgn_ip_mcp = np.sign(coords_IP[2] - coords_MCP[2])
        sgn_MCP_CMC = np.sign(coords_MCP[2] - coords_CMC[2])
        sgn_vec = np.array([sgn_tip_ip, sgn_ip_mcp, sgn_MCP_CMC])

        target = np.concatenate((unit_CMC, unit_MCP, unit_IP, unit_TIP, dot_vec, sgn_vec, ratio_vec))
        targets[finger].append(target)

def normalizeTargets():
    ratio1_dict = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: []
    }

    ratio2_dict = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: []
    }

    for finger in range(5):
        for target in targets[finger]:
            ratios = target[-2:]
            ratio1 = ratios[0]
            ratio2 = ratios[1]

            ratio1_dict[finger].append(ratio1)
            ratio2_dict[finger].append(ratio2)
    
    for finger in range(5):
        mean_ratio1 = np.mean(np.log(ratio1_dict[finger]))
        mean_ratio2 = np.mean(np.log(ratio2_dict[finger]))
        std_ratio1 = np.std(np.log(ratio1_dict[finger]))
        std_ratio2 = np.std(np.log(ratio2_dict[finger]))
        ratio1_means.append(mean_ratio1)
        ratio1_stds.append(std_ratio1)
        ratio2_means.append(mean_ratio2)
        ratio2_stds.append(std_ratio2)
    
    for finger in range(5):
        for target in targets[finger]:
            ratios = target[-2:]
            ratio1 = ratios[0]
            ratio2 = ratios[1]

            new_ratio1 = (np.log(ratio1) - ratio1_means[finger]) / ratio1_stds[finger]
            new_ratio2 = (np.log(ratio2) - ratio2_means[finger]) / ratio2_stds[finger]
            ratios = [new_ratio1, new_ratio2]

def getFinalTargetsVec():
    for i in range(len(targets[0])):
        target_vec = np.concatenate((
            targets[0][i],
            targets[1][i],
            targets[2][i],
            targets[3][i],
            targets[4][i]
        ))
        target_vecs.append(target_vec)

def saveInfo():
    x_coords_left_mean = np.mean(x_coords_left, axis = 0).tolist()
    x_coords_left_std = np.std(x_coords_left, axis = 0).tolist()
    x_coord_left_stats = [x_coords_left_mean, x_coords_left_std]

    y_coords_left_mean = np.mean(y_coords_left, axis = 0).tolist()
    y_coords_left_std = np.std(y_coords_left, axis = 0).tolist()
    y_coord_left_stats = [y_coords_left_mean, y_coords_left_std]

    x_coords_right_mean = np.mean(x_coords_right, axis = 0).tolist()
    x_coords_right_std = np.std(x_coords_right, axis = 0).tolist()
    x_coord_right_stats = [x_coords_right_mean, x_coords_right_std]

    y_coords_right_mean = np.mean(y_coords_right, axis = 0).tolist()
    y_coords_right_std = np.std(y_coords_right, axis = 0).tolist()
    y_coord_right_stats = [y_coords_right_mean, y_coords_right_std]

    standardized_stats["x_train_stand_left"] = x_coord_left_stats
    standardized_stats["y_train_stand_left"] =  y_coord_left_stats
    standardized_stats["x_train_stand_right"] = x_coord_right_stats
    standardized_stats["y_train_stand_right"] = y_coord_right_stats
    standardized_stats["ratio1"] = list(zip(ratio1_means, ratio1_stds))
    standardized_stats["ratio2"] = list(zip(ratio2_means, ratio2_stds))

    with open(str(DIR / "standstats.json"), "w") as f:
        json.dump(standardized_stats, f, indent = 4)

def getInputTarget():
    standardized_x_left = zscore(np.array(x_coords_left), axis = 0)
    standardized_y_left = zscore(np.array(y_coords_left), axis = 0)
    standardized_x_right = zscore(np.array(x_coords_right), axis = 0)
    standardized_y_right = zscore(np.array(y_coords_right), axis = 0)

    remaining = 0
    for i in range(standardized_x_left.shape[0]):
        input_vec = torch.tensor(np.concatenate((standardized_x_left[i], standardized_y_left[i], standardized_x_right[i], standardized_y_right[i])), dtype = torch.float)
        target_vec = torch.tensor(target_vecs[i], dtype = torch.float)
        dir = str(DIR / "Training" / (str(i).zfill(5) + ".pt"))
        torch.save((input_vec, target_vec), dir)
    
def appendData():
    error_dirs_left = []
    error_dirs_right = []
    count = 1
    for bg in range(1, BG_COUNT + 1):
        for pose in POSES:
            for n in range(1200):
                left_img_dir = DIR / f"B{bg}{pose}" / "Left" / f"BB_left_{n}.png"
                right_img_dir = DIR / f"B{bg}{pose}" / "Right" / f"BB_right_{n}.png"
                
                left_img = cv2.imread(str(left_img_dir), cv2.IMREAD_COLOR)
                right_img = cv2.imread(str(right_img_dir), cv2.IMREAD_COLOR)

                try:
                    kps = rtmpose.get_keypoints(left_img, right_img)
                except: 
                    error_dirs_left.append(left_img_dir)
                    error_dirs_right.append(right_img_dir)
                    
                left_kps = kps[0][0]
                right_kps = kps[0][1]

                left_x = []
                left_y = []
                right_x = []
                right_y = []

                for i in range(21):
                    x_left = left_kps[i][0]
                    y_left = left_kps[i][1]
                    x_right = right_kps[i][0]
                    y_right = right_kps[i][1]
                    
                    x_norm_left, y_norm_left = normalize(x_left, y_left)
                    x_norm_right, y_norm_right = normalize(x_right, y_right)
                    left_x.append(x_norm_left)
                    left_y.append(y_norm_left)
                    right_x.append(x_norm_right)
                    right_y.append(y_norm_right)

                x_coords_left.append(left_x)
                y_coords_left.append(left_y)
                x_coords_right.append(right_x)
                y_coords_right.append(right_y)

                coords_3d = loadCoords(bg, pose, n)
                generateTarget(coords_3d)

                print(count)
                count += 1

appendData()    
normalizeTargets()
getFinalTargetsVec()
saveInfo()
getInputTarget()