import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/miket/Documents/Hand-Tracking-2/Keypoints")
from keypointdetection import ANGLE_JOINTS, HAND_SKELETON  # type: ignore

SANE_COORD_BOUND = 500.0
BONE_LENGTH_BOUND = 50.0

BONE_PAIRS = []
for finger in range(5):
    mcp = 1 + 4 * finger
    pip = 2 + 4 * finger
    dip = 3 + 4 * finger
    tip = 4 + 4 * finger
    BONE_PAIRS += [(0, mcp), (mcp, pip), (pip, dip), (dip, tip)]


def isCatastrophic(coords: np.ndarray):
    if np.abs(coords).max() > SANE_COORD_BOUND:
        return True
    for a, b in BONE_PAIRS:
        if np.linalg.norm(coords[a] - coords[b]) > BONE_LENGTH_BOUND:
            return True
    return False


def boneLengthError(coords, target):
    errs = []
    for parent, child in HAND_SKELETON:
        len_pred = torch.linalg.norm(coords[child] - coords[parent])
        len_target = torch.linalg.norm(target[child] - target[parent])
        errs.append((len_target - len_pred) ** 2)
    return torch.stack(errs).mean().item()


def boneDirError(coords, target):
    errs = []
    for parent, child in HAND_SKELETON:
        bone_pred = torch.nn.functional.normalize(
            coords[child] - coords[parent], dim=-1
        )
        bone_target = torch.nn.functional.normalize(
            target[child] - target[parent], dim=-1
        )
        errs.append(((bone_pred - bone_target) ** 2).sum())
    return torch.stack(errs).mean().item()


def angleError(coords, target):
    errs = []
    for parent, joint, child in ANGLE_JOINTS:
        p1 = coords[parent] - coords[joint]
        p2 = coords[child] - coords[joint]
        t1 = target[parent] - target[joint]
        t2 = target[child] - target[joint]

        cos_pred = torch.nn.functional.cosine_similarity(
            p1.unsqueeze(0), p2.unsqueeze(0)
        )
        cos_target = torch.nn.functional.cosine_similarity(
            t1.unsqueeze(0), t2.unsqueeze(0)
        )
        errs.append((cos_pred - cos_target) ** 2)
    return torch.stack(errs).mean().item()


def summarize(name: str, arr: np.ndarray, unit: str = ""):
    print(f"\n--- {name} (n={len(arr)}) ---")
    if len(arr) == 0:
        print("  no samples")
        return
    print(f"  mean:   {arr.mean():.4f} {unit}")
    print(f"  median: {np.median(arr):.4f} {unit}")
    print(f"  std:    {arr.std():.4f}")
    print(f"  min/max: {arr.min():.4f} / {arr.max():.4f} {unit}")
    for p in [75, 90, 95, 99]:
        print(f"  p{p}: {np.percentile(arr, p):.4f} {unit}")


def getAnatomyBaseline(data_dir: str):
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.pt"))

    anatomy_sum = []
    flagged_count = 0

    for pt in files:
        coords, _, _, coords_optim = torch.load(pt, weights_only=False)
        coords = coords.float()
        coords_optim = coords_optim.float()

        catastrophic = isCatastrophic(coords.numpy()) or isCatastrophic(
            coords_optim.numpy()
        )
        if catastrophic:
            flagged_count += 1
            continue  # skip catastrophic samples entirely - same filter as baseline.py

        bl = boneLengthError(coords, coords_optim)
        bd = boneDirError(coords, coords_optim)
        ang = angleError(coords, coords_optim)

        anatomy_sum.append(bl + bd + ang)

    print(f"Total samples: {len(files)}")
    print(
        f"Flagged as catastrophic (excluded): {flagged_count} ({100 * flagged_count / len(files):.1f}%)"
    )

    summarize(
        "Anatomy loss (boneLength + boneDir + angle, combined)", np.array(anatomy_sum)
    )


if __name__ == "__main__":
    print("Training:")
    getAnatomyBaseline("/home/miket/Documents/StereoDataset/Training")

    print("\nValidation:")
    getAnatomyBaseline("/home/miket/Documents/StereoDataset/Validation")

    print("\nTesting")
    getAnatomyBaseline("/home/miket/Documents/StereoDataset/Testing")
