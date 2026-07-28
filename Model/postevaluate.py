import json
from pathlib import Path

import torch
from model import AnatomyModel
from posttrain import Loss
from posttrainer import Trainer
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

with open("/home/miket/Documents/Hand-Tracking-2/Model/decpostconfigs.json", "r") as f:
    configs = json.load(f)


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

test_dataset = torch.load(Path(configs["stereo_data_dir"]) / "Testing" / "dataset.pt")
test_loader = DataLoader(
    dataset=test_dataset,
    num_workers=configs["num_workers"],
    batch_size=configs["batch_size"],
    drop_last=configs["drop_last"],
    collate_fn=collate,
)
standardize = Trainer().standardize
criterion = Loss(configs["delta"]).criterion


def evalModel():
    test_loss = 0
    test_dist = 0

    with torch.inference_mode():
        for batch, target, coords_proj in test_loader:
            batch = batch.to(device)
            target = target.to(device)
            coords_proj = coords_proj.to(device)

            features = batch.x.to(device, dtype=torch.float32)
            features = standardize(features)
            edge_index = batch.edge_index.to(device)
            b = batch.batch.to(device)

            errors = model(features, edge_index, b)
            errors = errors.view(errors.size(0), 21, 3)
            scale = torch.linalg.norm(coords_proj[:, 9] - coords_proj[:, 0], dim=1)
            pred = coords_proj + (scale[:, None, None] * errors)

            loss = criterion(pred, target)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    test_dist /= len(test_loader)
    return test_loss, test_dist


evaluation = evalModel()
print(f"Test Loss {evaluation[0]: .6f} | Test Dist Loss {evaluation[1]: .6f}")
