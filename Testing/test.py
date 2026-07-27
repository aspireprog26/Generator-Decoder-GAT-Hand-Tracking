"""
Computes baseline decoder error as mean Euclidean distance (cm) between
the raw triangulated coordinate (`coords`) and the optimized ground truth
(`coords_optim`), with zero model correction applied - both:

  1. Over ALL samples (unfiltered)
  2. Excluding samples flagged as catastrophic triangulation failures

This uses torch.linalg.norm(dim=-1) per joint, then averages across
joints and samples - the same metric your Trainer now reports for
decoder_val_loss, so this is a directly comparable "do nothing" number.

No files are modified or deleted - this is read-only, for comparison.

Usage:
    python baseline_check_distance.py /home/miket/Documents/StereoDataset/Validation
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Same bounds as filter_dataset.py - keep in sync if you've tuned these.
SANE_COORD_BOUND = 500.0
BONE_LENGTH_BOUND = 50.0

BONE_PAIRS = []
for finger in range(5):
    mcp = 1 + 4 * finger
    pip = 2 + 4 * finger
    dip = 3 + 4 * finger
    tip = 4 + 4 * finger
    BONE_PAIRS += [(0, mcp), (mcp, pip), (pip, dip), (dip, tip)]


def is_catastrophic(coords: np.ndarray) -> bool:
    if np.abs(coords).max() > SANE_COORD_BOUND:
        return True
    for a, b in BONE_PAIRS:
        if np.linalg.norm(coords[a] - coords[b]) > BONE_LENGTH_BOUND:
            return True
    return False


def summarize(name: str, arr: np.ndarray):
    print(f"\n--- {name} (n={len(arr)}) ---")
    if len(arr) == 0:
        print("  no samples")
        return
    print(f"  mean:   {arr.mean():.4f} cm")
    print(f"  median: {np.median(arr):.4f} cm")
    print(f"  std:    {arr.std():.4f}")
    print(f"  min/max: {arr.min():.4f} / {arr.max():.4f} cm")
    for p in [75, 90, 95, 99]:
        print(f"  p{p}: {np.percentile(arr, p):.4f} cm")


def main(data_dir: str):
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.pt"))

    all_dist = []
    clean_dist = []
    flagged_count = 0

    for pt in files:
        coords, _, _, coords_optim = torch.load(pt, weights_only=False)
        coords = coords.float()
        coords_optim = coords_optim.float()

        # Per-joint Euclidean distance, then mean across all 21 joints
        # for this single sample - matches the Trainer's per-batch metric.
        per_joint_dist = torch.linalg.norm(coords - coords_optim, dim=-1)  # (21,)
        sample_mean_dist = per_joint_dist.mean().item()
        all_dist.append(sample_mean_dist)

        catastrophic = is_catastrophic(coords.numpy()) or is_catastrophic(
            coords_optim.numpy()
        )
        if catastrophic:
            flagged_count += 1
        else:
            clean_dist.append(sample_mean_dist)

    all_dist = np.array(all_dist)
    clean_dist = np.array(clean_dist)

    print(f"Total samples: {len(files)}")
    print(
        f"Flagged as catastrophic: {flagged_count} ({100 * flagged_count / len(files):.1f}%)"
    )

    summarize("ALL SAMPLES (unfiltered) - mean joint distance", all_dist)
    summarize("EXCLUDING FLAGGED SAMPLES - mean joint distance", clean_dist)


if __name__ == "__main__":
    main("/home/miket/Documents/StereoDataset/Training")
