import json
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from trainer import Trainer
from model import AnatomyRegression
from dataset import StereoHandDataset
from torch.utils.data import DataLoader

configs = {
    "lr": 1e-4,
    "batch_size": 32,
    "input_size": 84,
    "hidden_size": 128,
    "output_size": 100,
    "num_workers": 2,
    "num_epochs": 500,
    "weight_decay": 1e-3,
    "model_dir": "/home/mrtcloud-1/Documents/Hand-Tracking-2/Anatomy",
    "model_name": "anatomy.pth",
    "data_dir": "/home/mrtcloud-1/Documents/StereoDataset",
    "es_patience": 5,
    "es_thresh": 1e-4,
    "shuffle": True,
    "drop_last": False
}

with open((Path(configs["model_dir"]) / "configs.json"), "w") as f:
    json.dump(configs, f, indent = 4)

train_dataset = StereoHandDataset(configs["data_dir"], "train")
train_loader = DataLoader(
    dataset = train_dataset,
    num_workers = configs["num_workers"],
    batch_size = configs["batch_size"],
    shuffle = configs["shuffle"],
    drop_last = configs["drop_last"]
)

val_dataset = StereoHandDataset(configs["data_dir"], "val")
val_loader = DataLoader(
    dataset = val_dataset,
    num_workers = configs["num_workers"],
    batch_size = configs["batch_size"],
    shuffle = configs["shuffle"],
    drop_last = configs["drop_last"]
)

criterion = nn.MSELoss()
model = AnatomyRegression(configs["input_size"], configs["hidden_size"], configs["output_size"])
optimizer = optim.AdamW(model.parameters(), lr = configs["lr"], weight_decay = configs["weight_decay"])
trainer = Trainer(model, configs, train_loader, val_loader, criterion, optimizer)
trainer.train()