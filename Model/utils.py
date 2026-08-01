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


def procrustesAlign(
    X,  # raw 3D keypoint
    Y,  # MANO predicted 3D keypoints
    allow_reflection=False,
    allow_scaling=True,
    eps=1e-7,
):
    in_dtype = X.dtype

    # Use float32 for numerical stability
    X32 = X.float()
    Y32 = Y.float()

    B, _, D = X32.shape

    # Center point clouds
    X_mean = X32.mean(dim=1, keepdim=True)
    Y_mean = Y32.mean(dim=1, keepdim=True)

    X_c = X32 - X_mean
    Y_c = Y32 - Y_mean

    # Cross covariance
    M = torch.bmm(X_c.transpose(1, 2), Y_c)

    # Add regularization
    jitter = 1e-6
    M = M + jitter * torch.eye(M.shape[-1], device=M.device, dtype=M.dtype).unsqueeze(0)

    # SVD
    U, S, Vh = torch.linalg.svd(M)

    # Rotation
    if allow_reflection:
        R = torch.bmm(U, Vh)
        scale_num = S.sum(dim=-1)
    else:
        det = torch.linalg.det(torch.bmm(U, Vh))

        sign = torch.where(
            det < 0,
            -torch.ones_like(det),
            torch.ones_like(det),
        ).detach()

        Dmat = (
            torch.eye(
                D,
                device=X.device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .repeat(B, 1, 1)
        )

        Dmat[:, -1, -1] = sign
        R = torch.bmm(
            torch.bmm(U, Dmat),
            Vh,
        )
        scale_num = S.sum(dim=-1) - (sign < 0).to(S.dtype) * 2.0 * S[:, -1]

    # Scale
    if allow_scaling:
        var_X = (X_c**2).sum(dim=(1, 2))
        var_X = torch.clamp(var_X, min=eps)
        s = (scale_num / var_X)[:, None, None]
    else:
        s = torch.ones(
            (B, 1, 1),
            device=X.device,
            dtype=torch.float32,
        )

    # Apply alignment
    X_aligned = s * torch.bmm(X_c, R) + Y_mean
    return X_aligned.to(in_dtype)


class Stats:
    def __init__(self, device):
        decoder_stats = torch.load(
            Path(configs["stereo_data_dir"]) / "Training" / "Decoder" / "stats.pt",
            weights_only=False,
        )
        self.decoder_mean = decoder_stats[0].to(device)
        self.decoder_std = decoder_stats[1].to(device)

    def standardize(self, features):
        stand_feats = (features - self.decoder_mean) / self.decoder_std
        return stand_feats


def saveConfigs(cfgs, name):
    with open((Path(cfgs["model_dir"]) / f"{name}.json"), "w") as f:
        json.dump(cfgs, f, indent=4)
