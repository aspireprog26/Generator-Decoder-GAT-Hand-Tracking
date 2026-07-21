import torch
from pathlib import Path
from torch_geometric.data import Data

hand_edge_index = torch.tensor([
    [0, 1, 0, 5, 0, 9, 0, 13, 0, 17, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 9, 10, 10, 11, 11, 12, 13, 14, 14, 15, 15, 16, 17, 18, 18, 19, 19, 20],
    [1, 0, 5, 0, 9, 0, 13, 0, 17, 0, 2, 1, 3, 2, 4, 3, 6, 5, 7, 6, 8, 7, 10, 9, 11, 10, 12, 11, 14, 13, 15, 14, 16, 15, 18, 17, 19, 18, 20, 19]
])

def createGraphDataset(mode, type):
    dataset = []
    dir = "/home/mrtcloud-1/Documents/StereoSTBDataset" if type == "STB" else "/home/mrtcloud-1/Documents/StereoDataset"
    dir = Path(dir)

    if mode == "train":
        dir = dir / "Training"
    elif mode == "val":
        dir = dir / "Validation"
    else:
        dir = dir / "Testing"
    
    for path in sorted(dir.glob("*.pt")):
        if type == "STB":
            coords = torch.load(path)
            graph = Data(x = coords, edge_index = hand_edge_index)
            data_pt = graph
        else:
            coords_proj, normalized_coords, coords_optim = torch.load(path)
            graph = Data(x = normalized_coords, edge_index = hand_edge_index)
            data_pt = (graph, coords_optim, coords_proj)
        dataset.append(data_pt)
    torch.save(dataset, dir / "dataset.pt")

def createGraphs(type: str):
    createGraphDataset("train", type)
    createGraphDataset("val", type)
    createGraphDataset("test", type)

createGraphs("STB")
createGraphs("Stereo")