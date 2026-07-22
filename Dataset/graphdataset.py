import torch
from pathlib import Path
from torch_geometric.data import Data
from handedgeindex import hand_edge_index


def createGraphDataset(mode, type):
    dataset = []
    dir = (
        "/home/mrtcloud-1/Documents/StereoSTBDataset"
        if type == "STB"
        else "/home/mrtcloud-1/Documents/StereoDataset"
    )
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
            graph = Data(x=coords, edge_index=hand_edge_index)
            data_pt = graph

        else:
            coords_proj, normalized_coords, coords_optim = torch.load(path)
            graph = Data(x=normalized_coords, edge_index=hand_edge_index)
            data_pt = (graph, coords_optim, coords_proj)
        dataset.append(data_pt)
    torch.save(dataset, dir / "dataset.pt")


def createGraphs(type: str):
    createGraphDataset("train", type)
    createGraphDataset("val", type)
    createGraphDataset("test", type)


createGraphs("STB")
createGraphs("Stereo")
