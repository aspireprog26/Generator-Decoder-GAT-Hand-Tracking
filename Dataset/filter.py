"""
Filters catastrophic triangulation-failure samples out of the dataset.

A "catastrophic" sample is one where the raw triangulated coords or the
optimized coords are physically implausible for a hand (not just a badly
posed hand - an impossible one). We check two things:

  1. Absolute coordinate magnitude - a hand should be within some sane
     distance of the camera rig. Anything wildly outside that range means
     triangulation produced a near-degenerate point.
  2. Bone length consistency - real hand bones have roughly fixed lengths.
     If wrist-to-mcp or mcp-to-pip distances are absurd, the triangulated
     points aren't representing a real hand at all.

Adjust SANE_COORD_BOUND and BONE_LENGTH_BOUND to your actual capture setup
(units match whatever `coords`/`coords_optim` are stored in - looks like
cm x 100 based on project3D).

Usage:
    python filter_dataset.py /home/miket/Documents/StereoDataset/Training --dry-run
    python filter_dataset.py /home/miket/Documents/StereoDataset/Training
"""

import argparse
from pathlib import Path

import numpy as np
import torch

# once you've looked at a few borderline cases.
SANE_COORD_BOUND = 500.0  # max plausible |coordinate| value
BONE_LENGTH_BOUND = 50.0  # max plausible distance between adjacent joints

# Adjacent-joint pairs (wrist=0, then 4 joints per finger, order per your
# next_joint_idx convention: mcp, pip, dip, tip for each of 5 fingers)
BONE_PAIRS = []
for finger in range(5):
    mcp = 1 + 4 * finger
    pip = 2 + 4 * finger
    dip = 3 + 4 * finger
    tip = 4 + 4 * finger
    BONE_PAIRS += [(0, mcp), (mcp, pip), (pip, dip), (dip, tip)]


def is_catastrophic(coords: torch.Tensor) -> tuple[bool, str]:
    coords = coords.float().numpy()

    if np.abs(coords).max() > SANE_COORD_BOUND:
        return True, f"coord magnitude {np.abs(coords).max():.1f} > {SANE_COORD_BOUND}"

    for a, b in BONE_PAIRS:
        d = np.linalg.norm(coords[a] - coords[b])
        if d > BONE_LENGTH_BOUND:
            return True, f"bone ({a},{b}) length {d:.1f} > {BONE_LENGTH_BOUND}"

    return False, ""


def main(data_dir: str, dry_run: bool):
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.pt"))

    bad = []
    for pt in files:
        coords, _, _, coords_optim = torch.load(pt, weights_only=False)

        bad_raw, reason_raw = is_catastrophic(coords)
        bad_optim, reason_optim = is_catastrophic(coords_optim)

        if bad_raw or bad_optim:
            reason = reason_raw if bad_raw else reason_optim
            bad.append((pt, reason))

    print(f"Total samples: {len(files)}")
    print(
        f"Catastrophic samples found: {len(bad)} ({100 * len(bad) / len(files):.1f}%)"
    )
    print("\nFirst 20 flagged:")
    for pt, reason in bad[:20]:
        print(f"  {pt.name:20s} {reason}")

    if dry_run:
        print("\nDry run - no files deleted. Re-run without --dry-run to remove them.")
    else:
        for pt, _ in bad:
            pt.unlink()
        print(f"\nDeleted {len(bad)} files.")


if __name__ == "__main__":
    main("/home/miket/Documents/StereoDataset/Training", True)
    main("/home/miket/Documents/StereoDataset/Validation", True)
    main("/home/miket/Documents/StereoDataset/Testing", True)
