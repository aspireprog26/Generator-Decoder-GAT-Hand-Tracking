import cv2
import sys
import json
import torch
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from torch_geometric.data import Data

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
RT_CAM_FX_RIGHT = 589.8875

RT_CAM_CX_LEFT = 383.6427
RT_CAM_CY_LEFT = 306.1255
RT_CAM_CX_RIGHT = 383.7412
RT_CAM_CY_RIGHT = 288.4137
                            
# Accumulator lists for structured frame-by-frame storage
left_features_list = []   # Will store arrays of shape (21, 2)
right_features_list = []  # Will store arrays of shape (21, 2)

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
    "ratio1": [],
    "ratio2": []
}

rtmpose = kp.RTMPose(ENGINE)
tensor_count = 0 
data = {}

def rigidTransform(orig_coords):
    keypoints_bb = (ROTATION_MAT_BB @ orig_coords.T).T + TRANSLATION_VEC_BB     
    return keypoints_bb

def loadCoords(bg, pose, frame):
    dir = LABEL_DIR / f"B{bg}{pose}_BB.mat"
    coords = loadmat(str(dir))["handPara"]
    coords_transposed = np.transpose(coords, (2, 1, 0))     
    return coords_transposed[frame, ...]                    

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
        
        unit_CMC = coords_CMC / (np.linalg.norm(coords_CMC) + 1e-5)
        unit_MCP = coords_MCP / (np.linalg.norm(coords_MCP) + 1e-5)
        unit_IP = coords_IP / (np.linalg.norm(coords_IP) + 1e-5)
        unit_TIP = coords_TIP / (np.linalg.norm(coords_TIP) + 1e-5)

        cmc_dot_mcp = np.dot(unit_CMC, unit_MCP)
        mcp_dot_ip = np.dot(unit_MCP, unit_IP)
        ip_dot_tip = np.dot(unit_IP, unit_TIP)
        dot_vec = np.array([cmc_dot_mcp, mcp_dot_ip, ip_dot_tip])

        tip_ip_mag = np.linalg.norm(coords_TIP - coords_IP)
        ip_mcp_mag = np.linalg.norm(coords_IP - coords_MCP)
        mcp_cmc_mag = np.linalg.norm(coords_MCP - coords_CMC)

        ratio_1 = tip_ip_mag / (ip_mcp_mag + 1e-5)
        ratio_2 = ip_mcp_mag / (mcp_cmc_mag + 1e-5)
        ratio_vec = np.array([ratio_1, ratio_2])

        sgn_tip_ip = np.sign(coords_TIP[2] - coords_IP[2])
        sgn_ip_mcp = np.sign(coords_IP[2] - coords_MCP[2])
        sgn_MCP_CMC = np.sign(coords_MCP[2] - coords_CMC[2])
        sgn_vec = np.array([sgn_tip_ip, sgn_ip_mcp, sgn_MCP_CMC])

        target = np.concatenate((unit_CMC, unit_MCP, unit_IP, unit_TIP, dot_vec, sgn_vec, ratio_vec))
        targets[finger].append(target)

def normalizeTargets(mode):
    if mode == "train":
        ratio1_dict = {0: [], 1: [], 2: [], 3: [], 4: []}
        ratio2_dict = {0: [], 1: [], 2: [], 3: [], 4: []}

        for finger in range(5):
            for target in targets[finger]:
                ratios = target[-2:]
                ratio1 = ratios[0] + 1e-5
                ratio2 = ratios[1] + 1e-5

                ratio1_dict[finger].append(ratio1)
                ratio2_dict[finger].append(ratio2)
        
        for finger in range(5):
            mean_ratio1 = np.mean(np.log(ratio1_dict[finger]))
            mean_ratio2 = np.mean(np.log(ratio2_dict[finger]))
            std_ratio1 = np.std(np.log(ratio1_dict[finger]))
            std_ratio2 = np.std(np.log(ratio2_dict[finger]))

            ratio1_means.append(mean_ratio1)
            ratio1_stds.append(max(std_ratio1, 1e-5))
            ratio2_means.append(mean_ratio2)
            ratio2_stds.append(max(std_ratio2, 1e-5))
        
        for finger in range(5):
            for target in targets[finger]:
                ratios = target[-2:]
                ratio1 = ratios[0] + 1e-5
                ratio2 = ratios[1] + 1e-5

                new_ratio1 = (np.log(ratio1) - ratio1_means[finger]) / ratio1_stds[finger]
                new_ratio2 = (np.log(ratio2) - ratio2_means[finger]) / ratio2_stds[finger]
                target[-2:] = [new_ratio1, new_ratio2]
    else:
        ratio1_stats = data["ratio1"]
        ratio2_stats = data["ratio2"]
        for finger in range(5):
            for target in targets[finger]:
                ratios = np.log(target[-2:] + 1e-5)
                ratio1 = ratios[0]
                ratio2 = ratios[1]

                ratio1_mean = ratio1_stats[finger][0]
                ratio1_std = ratio1_stats[finger][1]
                ratio2_mean = ratio2_stats[finger][0]
                ratio2_std = ratio2_stats[finger][1]

                new_ratio1 = (ratio1 - ratio1_mean) / ratio1_std
                new_ratio2 = (ratio2 - ratio2_mean) / ratio2_std
                target[-2:] = [new_ratio1, new_ratio2]

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
    standardized_stats["ratio1"] = list(zip(ratio1_means, ratio1_stds))
    standardized_stats["ratio2"] = list(zip(ratio2_means, ratio2_stds))

    with open(str(DIR / "standstats.json"), "w") as f:
        json.dump(standardized_stats, f, indent = 4)

def getInputTarget(folder: str):
    for i in range(len(left_features_list)):
        # Construct cleanly organized coordinate blocks for both hands -> shape (2, 21, 4)
        structured_input = np.stack([left_features_list[i], right_features_list[i]], axis = 0)
        
        input_tensor = torch.tensor(structured_input, dtype = torch.float)
        target_vec = torch.tensor(target_vecs[i], dtype = torch.float)
        
        out_dir = str(DIR / folder / (str(i).zfill(5) + ".pt"))
        torch.save((input_tensor, target_vec), out_dir)

def loop(start, end):
    # Clear out lists before running subset loops
    left_features_list.clear()
    right_features_list.clear()
    for finger in targets:
        targets[finger].clear()
    target_vecs.clear()

    count = 0
    for bg in range(1, BG_COUNT + 1):
        for pose in POSES:
            for n in range(start, end):
                left_img_dir = DIR / f"B{bg}{pose}" / "Left" / f"BB_left_{n}.png"
                right_img_dir = DIR / f"B{bg}{pose}" / "Right" / f"BB_right_{n}.png"
                
                left_img = cv2.imread(str(left_img_dir), cv2.IMREAD_COLOR)
                right_img = cv2.imread(str(right_img_dir), cv2.IMREAD_COLOR)

                try:
                    kps = rtmpose.get_keypoints(left_img, right_img)
                    left_kps = kps[0][0]
                    right_kps = kps[0][1]
                except Exception:
                    print(f"Skipping background: {bg}, pose: {pose}, frame: {n}.")
                    continue

                left_joints = []
                right_joints = []
                raw_coords_left = []
                raw_coords_right = []

                for i in range(21):
                    x_norm_left, y_norm_left = normalize(left_kps[i][0], left_kps[i][1])
                    x_norm_right, y_norm_right = normalize(right_kps[i][0], right_kps[i][1])
                    
                    raw_coords_left.append([left_kps[i][0], left_kps[i][1]])
                    raw_coords_right.append([right_kps[i][0], right_kps[i][1]])
                    left_joints.append([x_norm_left, y_norm_left])
                    right_joints.append([x_norm_right, y_norm_right])

                # Convert to arrays -> shape (21, 2)
                arr_left = np.array(left_joints)
                arr_right = np.array(right_joints)
                raw_arr_left = np.array(raw_coords_left)
                raw_arr_right = np.array(raw_coords_right)

                # Make all joint coordinates relative to the frame's wrist position (Index 0) and range of thumb to pinky
                range_x_left = raw_arr_left[20][0] - raw_arr_left[4][0]
                range_x_right = raw_arr_right[20][0] - raw_arr_right[4][0]
                range_y_left = raw_arr_left[12][1] - raw_arr_left[0][1]
                range_y_right = raw_arr_right[12][1] - raw_arr_right[0][1]
                
                if range_x_left == 0 or range_x_right == 0 or range_y_left == 0 or range_y_right == 0:
                    print(f"Skipping background: {bg}, pose: {pose}, frame: {n}.")
                    continue

                rel_left = raw_arr_left - raw_arr_left[0]
                rel_left[:, 0] /= range_x_left
                rel_left[:, 1] /= range_y_left

                rel_right = raw_arr_right - raw_arr_right[0]
                rel_right[:, 0] /= range_x_right
                rel_right[:, 1] /= range_y_right

                features_left = np.concatenate([arr_left, rel_left], axis = 1)
                features_right = np.concatenate([arr_right, rel_right], axis = 1) 

                left_features_list.append(features_left)
                right_features_list.append(features_right)

                coords_3d = loadCoords(bg, pose, n)
                generateTarget(coords_3d)
                
                count += 1
                print(count)

def trainData():
    loop(0, 1200)    
    normalizeTargets("train")
    getFinalTargetsVec()
    saveInfo()
    getInputTarget("Training")

def valData():
    loop(1200, 1350)
    normalizeTargets("val")
    getFinalTargetsVec()
    getInputTarget("Validation")

def testData():
    loop(1350, 1500)
    normalizeTargets("test")
    getFinalTargetsVec()
    getInputTarget("Testing")

def openData():
    with open(str(DIR / "standstats.json"), "r") as f:
        global data
        data = json.load(f)

trainData()
openData()
valData()
testData()

# Your original edge connections remain exactly the same
hand_edge_index = torch.tensor([
    [0, 1, 0, 5, 0, 9, 0, 13, 0, 17, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 9, 10, 10, 11, 11, 12, 13, 14, 14, 15, 15, 16, 17, 18, 18, 19, 19, 20],
    [1, 0, 5, 0, 9, 0, 13, 0, 17, 0, 2, 1, 3, 2, 4, 3, 6, 5, 7, 6, 8, 7, 10, 9, 11, 10, 12, 11, 14, 13, 15, 14, 16, 15, 18, 17, 19, 18, 20, 19]
])

def createGraphDataset(mode):
    dataset = []
    stereo_dir = Path("/home/mrtcloud-1/Documents/StereoDataset/")
    
    if mode == "train":
        dir = stereo_dir / "Training"
    elif mode == "val":
        dir = stereo_dir / "Validation"
    else:
        dir = stereo_dir / "Testing"
    
    # Track existing .pt sample files, excluding any previously saved dataset.pt
    for path in sorted(dir.glob("*.pt")):
        if path.name == "dataset.pt":
            continue
            
        input_tensor, stereo_optim_left, stereo_optim_right, target = torch.load(path)
        
        # input_tensor is now structured as shape (2, 21, 3)
        # index 0 is left hand joint coordinate grid, index 1 is right hand
        features_left = input_tensor[0]   # Shape: (21, 3)
        features_right = input_tensor[1]  # Shape: (21, 3)
        
        # Wrap directly into PyTorch Geometric Data objects
        left_graph = Data(x = features_left, edge_index = hand_edge_index)
        right_graph = Data(x = features_right, edge_index = hand_edge_index)
        
        data_pt = (left_graph, right_graph, stereo_optim_left, stereo_optim_right, target)
        dataset.append(data_pt)
        
    torch.save(dataset, dir / "dataset.pt")
    print(f"Successfully processed and built dataset.pt for {mode} mode.")

if __name__ == "__main__":
    createGraphDataset("train")
    createGraphDataset("val")
    createGraphDataset("test")