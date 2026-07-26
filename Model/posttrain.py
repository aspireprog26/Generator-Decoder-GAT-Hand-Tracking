import json
from pathlib import Path

import torch
from model import AnatomyModel
from posttrainer import Trainer
from pretrain import saveConfigs
from torch import nn, optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

with open("/home/miket/Documents/Hand-Tracking-2/Model/decpreconfigs.json", "r") as f:
    post_configs = json.load(f)


def collate(batch):
    graphs, targets, raw_coords = zip(*batch)
    batch = Batch.from_data_list(list(graphs))
    targets = torch.stack(targets, dim=0).float()
    raw_coords = torch.stack(raw_coords, dim=0).float()
    return (batch, targets, raw_coords)


train_dataset = torch.load(
    Path(post_configs["post_data_dir"]) / "Training" / "dataset.pt"
)
train_loader = DataLoader(
    dataset=train_dataset,
    num_workers=post_configs["num_workers"],
    batch_size=post_configs["batch_size"],
    shuffle=True,
    drop_last=post_configs["drop_last"],
    collate_fn=collate,
)

val_dataset = torch.load(
    Path(post_configs["post_data_dir"]) / "Validation" / "dataset.pt"
)
val_loader = DataLoader(
    dataset=val_dataset,
    num_workers=post_configs["num_workers"],
    batch_size=post_configs["batch_size"],
    drop_last=post_configs["drop_last"],
    collate_fn=collate,
)

criterion = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnatomyModel(
    post_configs["input_size"],
    post_configs["decoder_hidden_size"],
    post_configs["decoder_output_size"],
    post_configs["decoder_dropout"],
    generator=False,
).to(device)

weights = torch.load(
    (Path(post_configs["model_dir"]) / post_configs["model_name"]),
    weights_only=True,
    map_location=device,
)
model.load_state_dict(weights)

optimizer = optim.AdamW(
    model.parameters(),
    lr=post_configs["decoder_lr"],
    weight_decay=post_configs["weight_decay"],
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    min_lr=1e-6,
    factor=post_configs["scheduler_factor"],
    patience=post_configs["scheduler_patience"],
    threshold=post_configs["es_thresh"],
)

trainer = Trainer(
    model, post_configs, train_loader, val_loader, criterion, optimizer, scheduler
)
trainer.train()
saveConfigs(post_configs, "decpostconfigs")
