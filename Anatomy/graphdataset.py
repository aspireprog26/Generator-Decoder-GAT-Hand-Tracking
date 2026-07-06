import torch
from pathlib import Path
from torch_geometric.data import Data

hand_edge_index = torch.tensor([
    [0, 1, 0, 5, 0, 9, 0, 13, 0, 17, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 9, 10, 10, 11, 11, 12, 13, 14, 14, 15, 15, 16, 17, 18, 18, 19, 19, 20],
    [1, 0, 5, 0, 9, 0, 13, 0, 17, 0, 2, 1, 3, 2, 4, 3, 6, 5, 7, 6, 8, 7, 10, 9, 11, 10, 12, 11, 14, 13, 15, 14, 16, 15, 18, 17, 19, 18, 20, 19]
])

def createGraphDataset(mode):
    dataset = []
    stereo_dir = Path("/home/mrtcloud-1/Documents/StereoDataset/")
    if mode == "train":
        dir = stereo_dir / "Training"
    elif mode == "val":
        dir = stereo_dir / "Validation"
    else:
        dir = stereo_dir / "Testing"
    
    for path in sorted(dir.glob("*.pt")):
        input, target = torch.load(path)
        
        x_left, y_left, x_right, y_right = torch.chunk(input, 4, dim = -1)
        coords_left = torch.stack([x_left, y_left], dim = -1)  
        coords_right = torch.stack([x_right, y_right], dim = -1)
        
        left_graph = Data(x = coords_left, edge_index = hand_edge_index)
        right_graph = Data(x = coords_right, edge_index = hand_edge_index)
        data_pt = (left_graph, right_graph, target)
        dataset.append(data_pt)
    torch.save(dataset, dir / "dataset.pt")
    

createGraphDataset("train")
createGraphDataset("val")
createGraphDataset("test")