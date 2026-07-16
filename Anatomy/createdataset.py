import numpy as np
from pathlib import Path

class Dataset:
    def __init__(self):
        self.types = ["Clean", "Noisy"]
        self.data_dir = Path("C:\Users\Test\Documents\StereoDataset")
    
    def normalizeStereo(self):
        for type in self.types:
            for sample in (self.data_dir / type).glob("*.npy"):
                raw_coords, optimized_coords = np.load(sample)
                
    def normalizeSTB(self):
        None