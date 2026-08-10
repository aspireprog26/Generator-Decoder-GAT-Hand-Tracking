import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from manotrainer import Trainer as MANOTrainer
from model import AnatomyModel, MANOModel
from posttrain import Loss
from pretrainer import Trainer as PreTrainer
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from utils import Stats, procrustesAlign, stereoTransform

sys.path.insert(0, "/home/miket/Documents/Hand-Tracking-2/Keypoints")
sys.path.insert(0, "/home/miket/Documents/Hand-Tracking-2/Dataset")

from handedgeindex import hand_edge_index  # type: ignore
from keypointdetection import HAND_SKELETON, MediaPipe  # type: ignore

sample_eval = True

with open("/home/miket/Documents/Hand-Tracking-2/Model/decpostconfigs.json", "r") as f:
    dec_post_configs = json.load(f)

with open("/home/miket/Documents/Hand-Tracking-2/Model/manoconfigs.json", "r") as f:
    mano_configs = json.load(f)


def collate(batch):
    graphs, targets, coords_proj = zip(*batch)
    batch = Batch.from_data_list(list(graphs))
    targets = torch.stack(targets, dim=0).float()
    coords_proj = torch.stack(coords_proj, dim=0).float()
    return (batch, targets, coords_proj)


def plot(ax, points3D, orig=True):
    ax.zaxis.set_inverted(True)
    ax.view_init(elev=220, azim=130, roll=0)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.scatter(
        points3D[:, 0],
        points3D[:, 1],
        points3D[:, 2],
        color=(196 / 255, 12 / 255, 27 / 255),
        s=15,
    )

    for start, end in HAND_SKELETON:
        ax.plot(
            [points3D[start, 0], points3D[end, 0]],
            [points3D[start, 1], points3D[end, 1]],
            [points3D[start, 2], points3D[end, 2]],
            "b-",
        )

    ax.set_title(
        "Raw 3D Projected Hand Keypoints"
        if orig
        else "Corrected 3D Projected Hand Keypoints"
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

joint_predictor = AnatomyModel(
    dec_post_configs["input_size"],
    dec_post_configs["decoder_hidden_size"],
    dec_post_configs["decoder_hidden1"],
    dec_post_configs["decoder_output_size"],
    dec_post_configs["decoder_dropout"],
    generator=False,
).to(device)

joint_weights = torch.load(
    (Path(dec_post_configs["model_dir"]) / f"{dec_post_configs['decoder_model_name']}"),
    weights_only=True,
    map_location=device,
)
joint_predictor.load_state_dict(joint_weights)
joint_predictor.eval()

mano_model = MANOModel(
    mano_configs["input_size"],
    mano_configs["hidden_size"],
    mano_configs["hidden1"],
    mano_configs["dropout"],
    mano_configs["mano_root"],
    mano_configs["ncomps"],
).to(device)

mano_weights = torch.load(
    (Path(mano_configs["model_dir"]) / f"{mano_configs['model_name']}"),
    weights_only=True,
    map_location=device,
)
mano_model.load_state_dict(mano_weights)
mano_model.eval()


test_dataset = torch.load(
    Path(dec_post_configs["stereo_data_dir"]) / "Testing" / "Decoder" / "dataset.pt",
    weights_only=False,
)
test_loader = DataLoader(
    dataset=test_dataset,
    num_workers=dec_post_configs["num_workers"],
    batch_size=dec_post_configs["batch_size"],
    drop_last=dec_post_configs["drop_last"],
    collate_fn=collate,
)

standardize = Stats(device).standardize
feats = PreTrainer(
    dec_post_configs,
    None,
    None,
    joint_predictor,
    joint_predictor,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
).features

mano_feats = MANOTrainer(
    mano_model, joint_predictor, mano_configs, None, None, None, None, None, device
).getNewFeatures

criterion = Loss(
    dec_post_configs["delta1"],
    dec_post_configs["delta2"],
    dec_post_configs["w1"],
    dec_post_configs["w2"],
    dec_post_configs["w3"],
    dec_post_configs["w4"],
).criterion


def evalModel(sample: Path):
    test_loss = 0
    test_dist = 0
    test_samples = 0
    avg_time = 0

    if sample is None:
        with torch.inference_mode():
            for batch, target, coords_proj in test_loader:
                batch = batch.to(device)
                target = target.to(device)
                coords_proj = coords_proj.to(device)

                features = batch.x.to(device, dtype=torch.float32)
                orig_features = features.reshape(batch.num_graphs, 21, 19)
                features = standardize(orig_features)
                edge_index = batch.edge_index.to(device)
                b = batch.batch.to(device)

                joint_pred_coords = joint_predictor(features, edge_index, b)
                updated_features = mano_feats(orig_features, joint_pred_coords)
                updated_features = standardize(updated_features, decoder=False)
                pred_coords = mano_model(updated_features, edge_index, b)
                pred_coords = procrustesAlign(pred_coords, joint_pred_coords)
                pred_coords = stereoTransform(pred_coords, coords_proj)[0]
                loss, dist = criterion(pred_coords, target)

                test_dist += dist.item() * batch.num_graphs
                test_loss += loss.item() * batch.num_graphs
                test_samples += batch.num_graphs

        test_loss /= test_samples
        test_dist /= test_samples
    else:
        pose = MediaPipe()
        image = cv2.imread(sample, cv2.IMREAD_COLOR)
        _, w = image.shape[:2]
        half = w // 2

        left = image[:, :half]
        right = image[:, half:]

        left_kps, _, _ = pose.get_keypoints(left)
        right_kps, _, _ = pose.get_keypoints(right)

        fs = cv2.FileStorage(
            "/home/miket/Documents/Hand-Tracking-2/Stereo/stereo.yml",
            cv2.FILE_STORAGE_READ,
        )

        P1 = fs.getNode("P1").mat()
        P2 = fs.getNode("P2").mat()
        K1 = fs.getNode("K1").mat()
        K2 = fs.getNode("K2").mat()
        R1 = fs.getNode("R1").mat()
        R2 = fs.getNode("R2").mat()
        dist1 = fs.getNode("dist1").mat()
        dist2 = fs.getNode("dist2").mat()
        fs.release()

        pts_left = np.asarray(left_kps, dtype=np.float32).reshape(-1, 1, 2)
        pts_right = np.asarray(right_kps, dtype=np.float32).reshape(-1, 1, 2)

        pts_left_rect = cv2.undistortPoints(pts_left, K1, dist1, R=R1, P=P1)
        pts_right_rect = cv2.undistortPoints(pts_right, K2, dist2, R=R2, P=P2)

        pts_left_rect = pts_left_rect.squeeze(1)
        pts_right_rect = pts_right_rect.squeeze(1)

        points4D = cv2.triangulatePoints(P1, P2, pts_left_rect.T, pts_right_rect.T)
        points3D = (points4D[:3] / points4D[3]).T * 100
        points3D_orig = np.squeeze(points3D)

        points3D = torch.tensor(points3D_orig).unsqueeze(0).to(device)
        left_kps = torch.tensor(left_kps).unsqueeze(0).to(device)
        right_kps = torch.tensor(right_kps).unsqueeze(0).to(device)

        raw_feat, coords_proj, _ = feats(points3D, left_kps, right_kps)
        raw_feat = raw_feat.reshape(1, 21, 19)
        feat = standardize(raw_feat)

        edge_index = hand_edge_index.to(device)
        batch = torch.zeros(21, dtype=torch.long, device=device)

        with torch.inference_mode():
            joint_pred_coords = joint_predictor(feat, edge_index, batch)
            updated_features = mano_feats(raw_feat, joint_pred_coords)
            updated_features = standardize(updated_features, decoder=False)
            pred_coords = mano_model(updated_features, edge_index, batch)
            pred_coords = procrustesAlign(pred_coords, joint_pred_coords)
            pred_coords = stereoTransform(pred_coords, coords_proj)[0]
            points3D_corr = pred_coords.squeeze(0).cpu().numpy()

        print(coords_proj)
        print(pred_coords)

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

        plot(ax1, points3D_orig, orig=True)
        plot(ax2, points3D_corr, orig=False)

        plt.tight_layout()
        plt.show()

        t0 = time.time()
        for _ in range(500):
            with torch.inference_mode():
                joint_pred_coords = joint_predictor(feat, edge_index, batch)
                updated_features = mano_feats(raw_feat, joint_pred_coords)
                updated_features = standardize(updated_features, decoder=False)
                pred_coords = mano_model(updated_features, edge_index, batch)
                pred_coords = procrustesAlign(pred_coords, joint_pred_coords)
                pred_coords = stereoTransform(pred_coords, coords_proj)[0]
                points3D_corr = pred_coords.squeeze(0).cpu().numpy()
        t1 = time.time()
        avg_time = (t1 - t0) / 500

    return test_loss, test_dist, avg_time


if not sample_eval:
    test_loss, test_dist, avg_time = evalModel(None)
    print(f"Test Anatomy Loss {test_loss: .6f} | Test Dist Loss {test_dist: .6f}")
else:
    _, _, avg_time = evalModel("/home/miket/Documents/StereoDataset/Noisy/2607.jpg")
    print(f"Average Time: {avg_time: .6f}")
