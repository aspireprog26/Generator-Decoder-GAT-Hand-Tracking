from pathlib import Path

import numpy as np
import torch

SANE_COORD_BOUND = 500.0  # max plausible |coordinate| value
BONE_LENGTH_BOUND = 50.0  # max plausible distance between adjacent joints
BONE_PAIRS = []

for finger in range(5):
    mcp = 1 + 4 * finger
    pip = 2 + 4 * finger
    dip = 3 + 4 * finger
    tip = 4 + 4 * finger
    BONE_PAIRS += [(0, mcp), (mcp, pip), (pip, dip), (dip, tip)]


def isCatastrophic(coords: torch.Tensor):
    coords = coords.float().numpy()
    if np.abs(coords).max() > SANE_COORD_BOUND:
        return True, f"coord magnitude {np.abs(coords).max():.1f} > {SANE_COORD_BOUND}"

    for a, b in BONE_PAIRS:
        d = np.linalg.norm(coords[a] - coords[b])
        if d > BONE_LENGTH_BOUND:
            return True, f"bone ({a},{b}) length {d:.1f} > {BONE_LENGTH_BOUND}"

    return False, ""


def filterFiles(data_dir: str):
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.pt"))

    bad = []
    for pt in files:
        coords, _, _, coords_optim = torch.load(pt, weights_only=False)

        bad_raw, reason_raw = isCatastrophic(coords)
        bad_optim, reason_optim = isCatastrophic(coords_optim)

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

    for pt, _ in bad:
        pt.unlink()
    print(f"\nDeleted {len(bad)} files.")


if __name__ == "__main__":
    print("Training:")
    filterFiles("/home/miket/Documents/StereoDataset/Training")

    print("\nValidation:")
    filterFiles("/home/miket/Documents/StereoDataset/Validation")

    print("\nTesting")
    filterFiles("/home/miket/Documents/StereoDataset/Testing")
