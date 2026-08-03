import json
from pathlib import Path

import torch

HAND_SKELETON = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

ANGLE_JOINTS = [
    (1, 2, 3),
    (2, 3, 4),
    (5, 6, 7),
    (6, 7, 8),
    (9, 10, 11),
    (10, 11, 12),
    (13, 14, 15),
    (14, 15, 16),
    (17, 18, 19),
    (18, 19, 20),
]

with open("/home/miket/Documents/Hand-Tracking-2/Model/decpreconfigs.json", "r") as f:
    configs = json.load(f)


class Stats:
    def __init__(self, device):
        decoder_stats = torch.load(
            Path(configs["stereo_data_dir"]) / "Training" / "Decoder" / "stats.pt",
            weights_only=False,
        )
        self.decoder_mean = decoder_stats[0].to(device, dtype=torch.float32)
        self.decoder_std = decoder_stats[1].to(device, dtype=torch.float32)

    def standardize(self, features):
        stand_feats = (features - self.decoder_mean) / self.decoder_std
        return stand_feats


def saveConfigs(cfgs, name):
    with open((Path(cfgs["model_dir"]) / f"{name}.json"), "w") as f:
        json.dump(cfgs, f, indent=4)


def placeAtReference(mano_joints, coords_proj, eps=1e-8):
    """
    mano_joints: (B, 21, 3) canonical, root-relative MANO output (joint 0 ~ origin)
    coords_proj: (B, 21, 3) noisy stereo reference, in real coordinate space

    Rigidly places mano_joints at coords_proj's wrist position and scale,
    replacing the need for Procrustes alignment.
    """

    mano_scale = torch.linalg.norm(
        mano_joints[:, 9] - mano_joints[:, 0], dim=-1, keepdim=True
    ).clamp_min(eps)
    mano_joints_unit = mano_joints / mano_scale.unsqueeze(-1)

    target_scale = torch.linalg.norm(
        coords_proj[:, 9] - coords_proj[:, 0], dim=-1, keepdim=True
    ).clamp_min(eps)
    wrist_pos = coords_proj[:, 0:1, :]

    return mano_joints_unit * target_scale.unsqueeze(-1) + wrist_pos
