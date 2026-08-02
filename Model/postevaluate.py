import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import AnatomyModel
from posttrain import Loss
from pretrainer import Trainer as PreTrainer
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from utils import Stats, placeAtReference  # CHANGED: was procrustesAlign

sys.path.insert(0, "/home/miket/Documents/Hand-Tracking-2/Keypoints")
sys.path.insert(0, "/home/miket/Documents/Hand-Tracking-2/Dataset")

from handedgeindex import hand_edge_index  # type: ignore
from keypointdetection import HAND_SKELETON, MediaPipe  # type: ignore

PALM = [0, 1, 5, 9, 13, 17]

sample_eval = True

with open("/home/miket/Documents/Hand-Tracking-2/Model/decpostconfigs.json", "r") as f:
    configs = json.load(f)


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
        "Raw 3D Projected Stereo Mapped Hand Keypoints"
        if orig
        else "Corrected 3D Projected Stereo Mapped Hand Keypoints"
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AnatomyModel(
    configs["input_size"],
    configs["decoder_hidden_size"],
    configs["decoder_hidden1"],
    configs["decoder_output_size"],
    configs["decoder_dropout"],
    mano_root=configs["mano_root"],
    generator=False,
    ncomps=configs["ncomps"],
).to(device)

weights = torch.load(
    (Path(configs["model_dir"]) / f"{configs['decoder_model_name']}"),
    weights_only=True,
    map_location=device,
)
model.load_state_dict(weights)
model.eval()

test_dataset = torch.load(
    Path(configs["stereo_data_dir"]) / "Testing" / "Decoder" / "dataset.pt",
    weights_only=False,
)
test_loader = DataLoader(
    dataset=test_dataset,
    num_workers=configs["num_workers"],
    batch_size=configs["batch_size"],
    drop_last=configs["drop_last"],
    collate_fn=collate,
)

standardize = Stats(device).standardize
feats = PreTrainer(
    configs,
    None,
    None,
    model,
    model,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
).features
criterion = Loss(
    configs["delta1"],
    configs["delta2"],
    configs["w1"],
    configs["w2"],
    configs["w3"],
    configs["w4"],
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
                features = features.reshape(batch.num_graphs, 21, 19)
                features = standardize(features)
                edge_index = batch.edge_index.to(device)
                b = batch.batch.to(device)

                pred_coords = model(features, edge_index, b)
                pred_coords = placeAtReference(pred_coords, coords_proj)
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

        feat, coords_proj, _ = feats(points3D, left_kps, right_kps)
        feat = feat.reshape(1, 21, 19)
        feat = standardize(feat)

        edge_index = hand_edge_index.to(device)
        batch = torch.zeros(21, dtype=torch.long, device=device)

        with torch.inference_mode():
            pred_coords = model(feat, edge_index, batch)
            pred_coords = placeAtReference(pred_coords, coords_proj)
            points3D_corr = pred_coords.squeeze(0).cpu().numpy()

        fig = plt.figure(figsize=(14, 6))

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

        plot(ax1, points3D_orig, orig=True)
        plot(ax2, points3D_corr, orig=False)

        t0 = time.time()
        for _ in range(200):
            with torch.inference_mode():
                pred_coords = model(feat, edge_index, batch)
                pred_coords = placeAtReference(pred_coords, coords_proj)
                points3D_corr = pred_coords.squeeze(0).cpu().numpy()
        t1 = time.time()
        avg_time = (t1 - t0) / 200

    return test_loss, test_dist, avg_time


if not sample_eval:
    test_loss, test_dist, avg_time = evalModel(None)
    print(f"Test Anatomy Loss {test_loss: .6f} | Test Dist Loss {test_dist: .6f}")
else:
    _, _, avg_time = evalModel("/home/miket/Documents/StereoDataset/Clean/0000.jpg")
    print(f"Average Time: {avg_time: .4f}")
