import json
from pathlib import Path

import torch
from model import AnatomyModel
from posttrainer import Trainer
from pretrain import saveConfigs
from torch import optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

with open("/home/miket/Documents/Hand-Tracking-2/Model/decpreconfigs.json", "r") as f:
    dec_post_configs = json.load(f)


def collate(batch):
    graphs, targets, raw_coords = zip(*batch)
    batch = Batch.from_data_list(list(graphs))
    targets = torch.stack(targets, dim=0).float()
    raw_coords = torch.stack(raw_coords, dim=0).float()
    return (batch, targets, raw_coords)


train_dataset = torch.load(
    Path(dec_post_configs["stereo_data_dir"]) / "Training" / "Decoder" / "dataset.pt"
)

val_dataset = torch.load(
    Path(dec_post_configs["stereo_data_dir"]) / "Validation" / "Decoder" / "dataset.pt"
)

train_loader = DataLoader(
    dataset=train_dataset,
    num_workers=dec_post_configs["num_workers"],
    batch_size=dec_post_configs["batch_size"],
    shuffle=True,
    drop_last=dec_post_configs["drop_last"],
    collate_fn=collate,
)
val_loader = DataLoader(
    dataset=val_dataset,
    num_workers=dec_post_configs["num_workers"],
    batch_size=dec_post_configs["batch_size"],
    drop_last=dec_post_configs["drop_last"],
    collate_fn=collate,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnatomyModel(
    dec_post_configs["input_size"],
    dec_post_configs["decoder_hidden_size"],
    dec_post_configs["decoder_output_size"],
    dec_post_configs["decoder_dropout"],
    generator=False,
).to(device)

weights = torch.load(
    (Path(dec_post_configs["model_dir"]) / dec_post_configs["model_name"]),
    weights_only=True,
    map_location=device,
)
model.load_state_dict(weights)

optimizer = optim.AdamW(
    model.parameters(),
    lr=dec_post_configs["decoder_lr"],
    weight_decay=dec_post_configs["weight_decay"],
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    min_lr=1e-6,
    factor=dec_post_configs["scheduler_factor"],
    patience=dec_post_configs["scheduler_patience"],
    threshold=dec_post_configs["es_thresh"],
)

trainer = Trainer(
    model, dec_post_configs, train_loader, val_loader, optimizer, scheduler, device
)
trainer.train()
saveConfigs(dec_post_configs, "decpostconfigs")
