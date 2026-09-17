import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import loadmat
from tqdm import tqdm

sys.path.insert(0, "/Hand-Tracking-2/Keypoints")

import keypointdetection as kp  # type: ignore
from constrain import OptimizeHands  # type: ignore


class FeatureExtractor:
    def __init__(self):
        next_idx = [0]
        for finger in range(5):
            pip = 2 + 4 * finger
            dip = 3 + 4 * finger
            tip = 4 + 4 * finger
            next_idx += [pip, dip, tip, tip]
        self.next_joint_idx = np.array(next_idx, dtype=np.int64)

    def features(self, coords: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        coords = coords.astype(np.float64)

        wrist = coords[0:1, :]
        scale = np.linalg.norm(coords[9] - coords[0])
        scale = max(scale, eps)

        coords_norm = (coords - wrist) / scale

        joint_norms = np.linalg.norm(coords, axis=-1)
        joint_norms = np.clip(joint_norms, eps, None)
        dir_vectors = coords / joint_norms[:, None]

        next_coords = coords[self.next_joint_idx]
        dist_to_next = np.linalg.norm(next_coords - coords, axis=-1) / scale

        features = np.concatenate(
            [coords_norm, dir_vectors, dist_to_next[:, None]], axis=-1
        )
        return features

    def getFeatures(
        self,
        coords: np.ndarray,
        coords_left: np.ndarray,
        coords_right: np.ndarray,
        eps: float = 1e-8,
    ):

        feats_3d = self.features(coords, eps)  # (21, 7)
        feats_left = self.features(coords_left, eps)  # (21, 5)
        feats_right = self.features(coords_right, eps)  # (21, 5)

        scale_left = np.linalg.norm(coords_left[9] - coords_left[0])
        scale_left = max(scale_left, eps)

        scale_right = np.linalg.norm(coords_right[9] - coords_left[0])
        scale_right = max(scale_right, eps)

        avg_scale = (scale_left + scale_right) / 2
        disparity = (coords_left - coords_right) / avg_scale  # (21, 2)

        features = np.concatenate(
            [feats_3d, feats_left, feats_right, disparity], axis=-1
        )  # (21, 19)
        return features


class DatasetOptimizer:
    def __init__(self, mp: bool):
        if mp:
            self.pose = kp.MediaPipe()

        self.data_dir = Path("/StereoDataset")
        self.stb_dir = Path("/StereoSTBDataset")
        self.label_dir = self.stb_dir / "labels"

        self.types = ["Clean", "Noisy"]
        self.poses = ["Counting", "Random"]

        self.bg_count = 6
        self.getFeatures = FeatureExtractor().getFeatures

    def openCalibration(self):
        fs = cv2.FileStorage(
            "/Stereo/stereo.yml",
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
        with tqdm(total=18000 - 1) as pbar:
            for img_type in self.types:
                for img in (self.data_dir / img_type).glob("*.jpg"):
                    try:
                        image = cv2.imread(img, cv2.IMREAD_COLOR)
                        if image is None:
                            continue

                        _, w = image.shape[:2]
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
                        pbar.update(1)
                    except Exception as e:  # noqa: BLE001
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

    def saveSTB(self, start, end, path):
        count = 0
        tot = 1500 * (end - start) * 2
        with tqdm(total=tot - 1) as pbar:
            for bg in range(start, end):
                for pose in self.poses:
                    for n in range(1500):
                        coords = torch.tensor(self.loadCoords(bg, pose, n))
                        left_path = (
                            self.stb_dir / f"B{bg}{pose}" / "Left" / f"BB_left_{n}.png"
                        )
                        right_path = (
                            self.stb_dir
                            / f"B{bg}{pose}"
                            / "Right"
                            / f"BB_right_{n}.png"
                        )

                        try:
                            left_img = cv2.imread(left_path, cv2.IMREAD_COLOR)
                            right_img = cv2.imread(right_path, cv2.IMREAD_COLOR)

                            left_kps = torch.tensor(self.pose.get_keypoints(left_img))
                            right_kps = torch.tensor(self.pose.get_keypoints(right_img))
                        except Exception as e:  # noqa: BLE001
                            print(e)
                            continue

                        data = (coords, left_kps, right_kps)
                        torch.save(data, path / f"{count}.pt")
                        count += 1
                        pbar.update(1)

    def stereoSave(self):
        count = 0
        with tqdm(total=6227 - 1) as pbar:
            for img_type in self.types:
                for pt in (self.data_dir / img_type).glob("*.npy"):
                    coords, coords_optim = np.load(pt)
                    try:
                        image = cv2.imread(
                            self.data_dir / img_type / f"{pt.stem}.jpg",
                            cv2.IMREAD_COLOR,
                        )

                        _, w = image.shape[:2]
                        half = w // 2

                        left = image[:, :half]
                        right = image[:, half:]

                        left_kps = self.pose.get_keypoints(left)
                        right_kps = self.pose.get_keypoints(right)
                    except Exception as e:  # noqa: BLE001
                        print(e)
                        continue

                    normalized_features_dec = self.getFeatures(
                        coords, left_kps, right_kps
                    )
                    normalized_features_gen = self.getFeatures(
                        coords_optim, left_kps, right_kps
                    )
                    data = (
                        torch.tensor(coords),
                        torch.tensor(normalized_features_dec),
                        torch.tensor(normalized_features_gen),
                        torch.tensor(coords_optim),
                    )
                    torch.save(
                        data, (self.data_dir / f"{img_type}Normalized" / f"{count}.pt")
                    )
                    count += 1
                    pbar.update(1)

    def createTrainTestVal(self):
        clean_len = 3548
        noisy_len = 2680

        clean_train_end = int(clean_len * 0.7)
        clean_val_end = int(clean_len * 0.85)

        noisy_train_end = int(noisy_len * 0.7)
        noisy_val_end = int(noisy_len * 0.85)

        glob_count = 0
        with tqdm(total=6227 - 1) as pbar:
            for img_type in self.types:
                count = 0
                train_end = clean_train_end if img_type == "Clean" else noisy_train_end
                val_end = clean_val_end if img_type == "Noisy" else noisy_val_end

                for pt in (self.data_dir / f"{img_type}Normalized").glob("*.pt"):
                    (
                        coords,
                        normalized_features_dec,
                        normalized_features_gen,
                        coords_optim,
                    ) = torch.load(pt)
                    data = (
                        coords,
                        normalized_features_dec,
                        normalized_features_gen,
                        coords_optim,
                    )

                    if count < train_end:
                        save_dir = self.data_dir / "Training"
                    elif count < val_end:
                        save_dir = self.data_dir / "Validation"
                    else:
                        save_dir = self.data_dir / "Testing"

                    torch.save(data, (save_dir / f"{glob_count}.pt"))
                    count += 1  # noqa: SIM113
                    glob_count += 1
                    pbar.update(1)

    def standardize(self, mode, decoder=True):
        data = []
        dir = self.data_dir / mode

        for pt in dir.glob("*.pt"):
            _, normalized_features_dec, normalized_features_gen, _ = torch.load(pt)
            normalized_features = (
                normalized_features_dec if decoder else normalized_features_gen
            )
            data.append(normalized_features)

        data = torch.stack(data, dim=0)
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0).clamp(1e-6)

        stats = (mean, std)
        dir = dir / "Decoder" if decoder else dir / "Generator"
        torch.save(stats, dir / "stats.pt")

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

        print("Standardizing Stereo Dataset.")
        self.standardize("Training")
        self.standardize("Training", decoder=False)
        print("Standardization Complete.")

    def removeOld(self):
        for type in self.types:
            for img in (self.data_dir / type).glob("*.npy"):
                save_path = self.data_dir / type / f"{img.stem}.npy"
                os.remove(save_path)


optimizer = DatasetOptimizer(mp=True)
optimizer.optimize()
optimizer.saveData()
