import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from scipy.io import loadmat

sys.path.insert(0, "/home/mrtcloud-1/Documents/Hand-Tracking-2/Keypoints")

import keypointdetection as kp  # type: ignore
from constrain import OptimizeHands


class DatasetOptimizer:
    def __init__(self, mp: bool):
        if mp:
            self.pose = kp.MediaPipe()

        self.data_dir = Path("/home/mrtcloud-1/Documents/StereoDataset")
        self.stb_dir = Path("/home/mrtcloud-1/Documents/StereoSTBDataset/")
        self.label_dir = self.stb_dir / "labels"

        self.types = ["Clean", "Noisy"]
        self.poses = ["Counting", "Random"]

        self.bg_count = 6

    def openCalibration(self):
        fs = cv2.FileStorage(
            "/home/mrtcloud-1/Documents/Hand-Tracking-2/Stereo/stereo.yml",
            cv2.FILE_STORAGE_READ,
        )

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
        pts_left = np.asarray(pts_left, dtype=np.float32).reshape(-1, 1, 2)
        pts_right = np.asarray(pts_right, dtype=np.float32).reshape(-1, 1, 2)

        pts_left_rect = cv2.undistortPoints(
            pts_left, self.K1, self.dist1, R=self.R1, P=self.P1
        )
        pts_right_rect = cv2.undistortPoints(
            pts_right, self.K2, self.dist2, R=self.R2, P=self.P2
        )

        # Flatten back to (N, 2)
        pts_left_rect = pts_left_rect.squeeze(1)
        pts_right_rect = pts_right_rect.squeeze(1)

        # Obtain 4D points and scale to 3D
        points4D = cv2.triangulatePoints(
            self.P1, self.P2, pts_left_rect.T, pts_right_rect.T
        )

        points3D = (points4D[:3] / points4D[3]).T * 100
        points3D = np.squeeze(points3D)

        return points3D

    def optimize(self):
        self.openCalibration()
        print("Starting Optimization Process.")

        count = 0
        for img_type in self.types:
            for img in (self.data_dir / img_type).glob("*.jpg"):
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

    def loadCoords(self, bg, pose, frame):
        dir = self.label_dir / f"B{bg}{pose}_BB.mat"
        coords = loadmat(dir)["handPara"]
        coords_transposed = np.transpose(
            coords, (2, 1, 0)
        )  # Turns into shape (1500, 21, 3)
        return coords_transposed[frame, ...]  # Returns shape (21, 3)

    def getFeatures(self, coords: np.ndarray, noise=None):
        scale = np.linalg.norm(coords[9] - coords[0])
        coords_proj = (coords - (scale * noise)) if noise is not None else coords
        scale = np.linalg.norm(coords_proj[9] - coords_proj[0])

        coords_proj_norm = (coords_proj - coords_proj[0]) / scale

        wrist_unit = coords_proj[0] / np.linalg.norm(coords_proj[0])
        wrist_node_length = 0
        wrist_vec = np.append(wrist_unit, wrist_node_length)

        coords_normalized = coords_proj_norm.tolist()
        coords_normalized[0].extend(wrist_vec.tolist())

        for finger in range(5):
            MCP = 1 + 4 * finger
            PIP = 2 + 4 * finger
            DIP = 3 + 4 * finger
            TIP = 4 + 4 * finger

            mcp_unit = coords_proj[MCP] / np.linalg.norm(coords_proj[MCP])
            mcp_node_length = (
                np.linalg.norm(coords_proj[PIP] - coords_proj[MCP]) / scale
            )
            mcp_vec = np.append(mcp_unit, mcp_node_length)

            pip_unit = coords_proj[PIP] / np.linalg.norm(coords_proj[PIP])
            pip_node_length = (
                np.linalg.norm(coords_proj[DIP] - coords_proj[PIP]) / scale
            )
            pip_vec = np.append(pip_unit, pip_node_length)

            dip_unit = coords_proj[DIP] / np.linalg.norm(coords_proj[DIP])
            dip_node_length = (
                np.linalg.norm(coords_proj[TIP] - coords_proj[DIP]) / scale
            )
            dip_vec = np.append(dip_unit, dip_node_length)

            tip_unit = coords_proj[TIP] / np.linalg.norm(coords_proj[TIP])
            tip_node_length = 0
            tip_vec = np.append(tip_unit, tip_node_length)

            coords_normalized[MCP].extend(mcp_vec.tolist())
            coords_normalized[PIP].extend(pip_vec.tolist())
            coords_normalized[DIP].extend(dip_vec.tolist())
            coords_normalized[TIP].extend(tip_vec.tolist())

        features = np.array(coords_normalized)
        return features, coords_proj, scale.item()

    def saveSTB(self, start, end, path):
        count = 0
        for bg in range(start, end):
            for pose in self.poses:
                for n in range(0, 1500):
                    coords = torch.tensor(self.loadCoords(bg, pose, n))
                    torch.save(coords, path / f"{count}.pt")
                    count += 1

    def stereoSave(self):
        count = 0
        for img_type in self.types:
            for pt in (self.data_dir / img_type).glob("*.npy"):
                coords_proj, coords_optim = np.load(pt)
                scale_opt = np.linalg.norm(coords_optim[9] - coords_optim[0])
                error = (coords_optim - coords_proj) / scale_opt
                normalized_coords = self.getFeatures(coords_proj)[0]
                data = (
                    torch.tensor(coords_proj),
                    torch.tensor(normalized_coords),
                    torch.tensor(coords_optim),
                    torch.tensor(error),
                )
                torch.save(
                    data, (self.data_dir / f"{img_type}Normalized" / f"{count}.pt")
                )
                count += 1

    def createTrainTestVal(self):
        clean_len = 3548
        noisy_len = 2680

        clean_train_end = int(clean_len * 0.7)
        clean_val_end = int(clean_len * 0.85)

        noisy_train_end = int(noisy_len * 0.7)
        noisy_val_end = int(noisy_len * 0.85)

        glob_count = 0
        for img_type in self.types:
            count = 0
            train_end = clean_train_end if img_type == "Clean" else noisy_train_end
            val_end = clean_val_end if img_type == "Noisy" else noisy_val_end

            for pt in (self.data_dir / f"{img_type}Normalized").glob("*.pt"):
                coords_proj, normalized_coords, coords_optim, error = torch.load(pt)
                data = (coords_proj, normalized_coords, coords_optim)

                if count < train_end:
                    save_dir = self.data_dir / "Training"
                elif count < val_end:
                    save_dir = self.data_dir / "Validation"
                else:
                    save_dir = self.data_dir / "Testing"

                torch.save(data, (save_dir / f"{glob_count}.pt"))
                count += 1
                glob_count += 1

    def saveData(self):
        print("Starting STB Dataset.")
        self.saveSTB(1, 5, self.stb_dir / "Training")
        self.saveSTB(5, 6, self.stb_dir / "Validation")
        self.saveSTB(6, 7, self.stb_dir / "Testing")
        print("STB Dataset Complete.")

        print("\nStarting Stereo Dataset.")
        self.stereoSave()
        self.createTrainTestVal()
        print("Stereo Dataset Complete.")

    def removeOld(self):
        for type in self.types:
            for img in (self.data_dir / type).glob("*.npy"):
                save_path = self.data_dir / type / f"{img.stem}.npy"
                os.remove(save_path)


"""
optimizer = DatasetOptimizer(mp = True)
optimizer.optimize()
optimizer.saveData()
"""
