import torch
from pathlib import Path
from torch.utils.data import Dataset

class StereoHandDataset(Dataset):
    def __init__(self, data_dir, mode):
        stereo_dir = Path(data_dir)
        if mode == "train":
            dir = stereo_dir / "Training"
        elif mode == "val":
            dir = stereo_dir / "Validation"
        else:
            dir = stereo_dir / "Testing"
        self.data = sorted(dir.glob("*.pt"))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input, target = torch.load(self.data[idx])
        return input, target