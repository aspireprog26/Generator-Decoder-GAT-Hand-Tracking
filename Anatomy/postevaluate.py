import json
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from model import AnatomyModel
from torch.utils.data import DataLoader

with open("/Users/michaeltoppin/Documents/Coding/Hand-Tracking-2/Anatomy/configs.json", "r") as f:
    configs = json.load(f)

def nmse(target, prediction):
    return np.mean((target - prediction) ** 2) / np.var(target)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnatomyModel(configs["input_size"], configs["hidden_size"], configs["output_size"], configs["dropout"]).to(device)
weights = torch.load((Path(configs["model_dir"]) / configs["model_name"]), weights_only = True, map_location = device)
model.load_state_dict(weights)
model.eval()

test_dataset = torch.load(Path(configs["post_data_dir"]) / "Testing" / "dataset.pt")
test_loader = DataLoader(
    dataset = test_dataset,
    num_workers = configs["num_workers"],
    batch_size = configs["batch_size"],
    drop_last = configs["drop_last"]
)

def evalModel():
    test_mse_loss = 0
    test_nmse_loss = 0
    total_samples = 0

    criterion = nn.MSELoss()
    with torch.inference_mode():
        for batch, target, raw_coords in test_loader:
            batch = batch.to(device)
            target = target.to(device)
            raw_coords = raw_coords.to(device)

            features = batch.x
            edge_index = batch.edge_index
            b = batch.batch

            error = model(features, edge_index, b)
            scale = torch.linalg.norm(raw_coords[9] - raw_coords[0])
            pred = raw_coords + (scale * error)
            loss = criterion(pred, target)

            test_mse_loss += loss.item()
            # test_nmse_loss += nmse(pred.cpu().numpy(), target.cpu().numpy()) * b.max().item()
            # total_samples += b.max().item()[0]

    avg_mse_loss = test_mse_loss / len(test_loader)
    # avg_nmse_loss = test_nmse_loss / total_samples
    return avg_mse_loss  # [avg_mse_loss, avg_nmse_loss]

evaluation = evalModel()
print(f"Average MSE Loss Per Batch: {evaluation: .5f}")
