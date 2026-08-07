from pathlib import Path

import torch
from Dataset.handedgeindex import hand_edge_index
from torch_geometric.data import Data


def createGraphDataset(mode, type, decoder: bool = True):
    dataset = []
    dir = (
        "/home/miket/Documents/StereoSTBDataset"
        if type == "STB"
        else "/home/miket/Documents/StereoDataset"
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
            coords, left_kps, right_kps = torch.load(path, weights_only=False)
            graph = Data(x=coords, edge_index=hand_edge_index)
            data_pt = (graph, left_kps, right_kps)
        else:
            (
                coords_proj,
                normalized_features_dec,
                normalized_features_gen,
                coords_optim,
            ) = torch.load(path, weights_only=False)
            graph = (
                Data(x=normalized_features_dec, edge_index=hand_edge_index)
                if decoder
                else Data(x=normalized_features_gen, edge_index=hand_edge_index)
            )
            data_pt = (graph, coords_optim, coords_proj)
        dataset.append(data_pt)

    if type != "STB" and decoder:
        dir = dir / "Decoder"
    elif type != "STB" and not decoder:
        dir = dir / "Generator"
    torch.save(dataset, dir / "dataset.pt")


def createGraphs(type: str, decoder=True):
    createGraphDataset("train", type, decoder)
    createGraphDataset("val", type, decoder)
    createGraphDataset("test", type, decoder)


print("Starting STB Graph Generation.")
createGraphs("STB")
print("STB Graph Generation Complete.")


print("Starting Stereo Graph Generation.")
createGraphs("Stereo")  # Create decoder dataset for post training
createGraphs("Stereo", decoder=False)  # Create generator dataset
print("Stereo Graph Generation Complete.")
