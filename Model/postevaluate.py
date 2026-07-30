import json
import sys
from pathlib import Path

import cv2
import matplotlib as plt
import numpy as np
import torch
from model import AnatomyModel
from posttrain import Loss
from Model.posttrainerreg import Trainer
from pretrainer import Trainer as PreTrainer
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

sys.path.insert(0, "/home/miket/Documents/Hand-Tracking-2/Keypoints")
sys.path.insert(0, "/home/miket/Documents/Hand-Tracking-2/Dataset/handedgeindex.py")
from handedgeindex import hand_edge_index  # type: ignore
from keypointdetection import HAND_SKELETON, MediaPipe  # type: ignore

sample_eval = False
with open("/home/miket/Documents/Hand-Tracking-2/Model/decpostconfigs.json", "r") as f:
    configs = json.load(f)


def collate(batch):
    graphs, targets, coords_proj = zip(*batch)
    batch = Batch.from_data_list(list(graphs))
    targets = torch.stack(targets, dim=0).float()
    coords_proj = torch.stack(coords_proj, dim=0).float()
    return (batch, targets, coords_proj)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnatomyModel(
    configs["input_size"],
    configs["decoder_hidden_size"],
    configs["decoder_output_size"],
    configs["decoder_dropout"],
    generator=False,
).to(device)

weights = torch.load(
    (Path(configs["model_dir"]) / configs["decoder_model_name"]),
    weights_only=True,
    map_location=device,
)

model.load_state_dict(weights)
model.eval()

test_dataset = torch.load(Path(configs["stereo_data_dir"]) / "Testing" / "dataset.pt")
test_loader = DataLoader(
    dataset=test_dataset,
    num_workers=configs["num_workers"],
    batch_size=configs["batch_size"],
    drop_last=configs["drop_last"],
    collate_fn=collate,
)

standardize = Trainer().standardize
feats = PreTrainer().features
criterion = Loss(configs["delta"]).criterion


def evalModel(sample: Path):
    test_loss = 0
    test_dist = 0

    if sample is not None:
        with torch.inference_mode():
            for batch, target, coords_proj in test_loader:
                batch = batch.to(device)
                target = target.to(device)
                coords_proj = coords_proj.to(device)

                features = batch.x.to(device, dtype=torch.float32)
                features = standardize(features)
                edge_index = batch.edge_index.to(device)
                b = batch.batch.to(device)

                errors = model(features, edge_index, b)
                errors = errors.view(errors.size(0), 21, 3)
                scale = torch.linalg.norm(coords_proj[:, 9] - coords_proj[:, 0], dim=1)
                pred = coords_proj + (scale[:, None, None] * errors)

                loss = criterion(pred, target)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        test_dist /= len(test_loader)
    else:
        pose = MediaPipe()
        image = cv2.imread(sample, cv2.IMREAD_COLOR)
        _, w = image.shape[:2]
        half = w // 2

        left = image[:, :half]
        right = image[:, half:]

        left_kps = torch.tensor(pose.get_keypoints(left))
        right_kps = torch.tensor(pose.get_keypoints(right))

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

        # Flatten back to (N, 2)
        pts_left_rect = pts_left_rect.squeeze(1)
        pts_right_rect = pts_right_rect.squeeze(1)

        # Obtain 4D points and scale to 3D
        points4D = cv2.triangulatePoints(P1, P2, pts_left_rect.T, pts_right_rect.T)
        points3D = (points4D[:3] / points4D[3]).T * 100
        points3D = np.squeeze(points3D)
        points3D = torch.tensor(points3D)

        feat, coords_proj, scale = feats(points3D, left_kps, right_kps)
        feat = standardize(feat)

        edge_index = hand_edge_index.to(device)
        batch = torch.zeros(21, dtype=torch.long, device=device)

        with torch.inference_mode():
            errors = model(features, edge_index, batch)  # (1, 63)
            errors = errors.view(1, 21, 3)

            scale = torch.linalg.norm(coords_proj[:, 9] - coords_proj[:, 0], dim=-1)
            pred_coords = coords_proj + (scale[:, None, None] * errors)
            points3D = pred_coords.squeeze(0).cpu().numpy()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

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
        plt.title("3D Mapped Hand Skeleton Keypoints (In Centimeters)")
        plt.show()

    return test_loss, test_dist


if sample_eval:
    evaluation = evalModel(None)
    print(f"Test Loss {evaluation[0]: .6f} | Test Dist Loss {evaluation[1]: .6f}")
else:
    evalModel("")
