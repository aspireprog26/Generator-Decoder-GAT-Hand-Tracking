import json
from pathlib import Path

import numpy as np
import torch
from model import AnatomyModel
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

with open("/home/miket/decconfigs.json", "r") as f:
    configs = json.load(f)


def nmse(prediction, target):
    return np.mean((target - prediction) ** 2) / np.var(target)


def collate(batch):
    graphs, targets, coords_proj = zip(*batch)
    batch = Batch.from_data_list(list(graphs))
    targets = torch.stack(targets, dim=0).float()
    coords_proj = torch.stack(coords_proj, dim=0).float()
    return (batch, targets, coords_proj)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnatomyModel(
    configs["input_size"],
    configs["decoder_hidden_size"],
    configs["decoder_output_size"],
    configs["decoder_dropout"],
    generator=False,
).to(device)

weights = torch.load(
    (Path(configs["model_dir"]) / configs["decoder_model_name"]),
    weights_only=True,
    map_location=device,
)

model.load_state_dict(weights)
model.eval()

test_dataset = torch.load(Path(configs["post_data_dir"]) / "Testing" / "dataset.pt")
test_loader = DataLoader(
    dataset=test_dataset,
    num_workers=configs["num_workers"],
    batch_size=configs["batch_size"],
    drop_last=configs["drop_last"],
    collate_fn=collate,
)


def evalModel():
    test_mse_loss = 0
    test_nmse_loss = 0
    total_samples = 0

    criterion = nn.MSELoss()
    with torch.inference_mode():
        for batch, target, coords_proj in test_loader:
            batch = batch.to(device)
            target = target.to(device)
            coords_proj = coords_proj.to(device)

            features = batch.x
            edge_index = batch.edge_index
            b = batch.batch

            error = model(features, edge_index, b)
            scale = torch.linalg.norm(coords_proj[:, 9] - coords_proj[:, 0], dim=1)
            pred = coords_proj + (scale[:, None, None] * error)
            loss = criterion(pred, target)

            test_mse_loss += loss.item()
            batch_size = target.size(0)
            test_nmse_loss += (
                nmse(pred.cpu().numpy(), target.cpu().numpy()) * batch_size
            )
            total_samples += batch_size

    avg_mse_loss = test_mse_loss / len(test_loader)
    avg_nmse_loss = test_nmse_loss / total_samples
    return (avg_mse_loss, avg_nmse_loss)


evaluation = evalModel()
print(
    f"Average MSE Loss Per Batch: {evaluation[0]: .5f} | Average NMSE Loss Per Sample: {evaluation[1]: .5f}"
)
