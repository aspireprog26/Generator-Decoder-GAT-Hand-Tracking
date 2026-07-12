import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from trainer import Trainer
from model import AnatomyModel
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

sys.path.insert(0, "/Users/michaeltoppin/Documents/Coding/Hand-Tracking-2/Model")
from Model.optimize import Losses

configs = {
    "lr": 1e-4,
    "batch_size": 32,
    "input_size": 4,
    "output_size": 100,
    "num_workers": 2,
    "num_epochs": 250,
    "weight_decay": 1e-2,
    "model_dir": "/home/mrtcloud-1/Documents/Hand-Tracking-2/Anatomy",
    "model_name": "anatomy.pth",
    "data_dir": "/home/mrtcloud-1/Documents/StereoDataset",
    "es_patience": 10,
    "es_thresh": 1e-4,
    "drop_last": False
}
 
def mulitloss(outputs):
    Non

def collate(batch):
    left_graphs, right_graphs, stereo_optim_left, stereo_optim_right, targets = zip(*batch)
    left_batch = Batch.from_data_list(list(left_graphs))
    right_batch = Batch.from_data_list(list(right_graphs))
    stereo_optim_left = torch.stack(stereo_optim_left, dim = 0).float()
    stereo_optim_right = torch.stack(stereo_optim_right, dim = 0).float()
    targets = torch.stack(targets, dim = 0).float()
    return left_batch, right_batch, targets

train_dataset = torch.load(Path(configs["data_dir"]) / "Training" / "dataset.pt")
train_loader = DataLoader(
    dataset = train_dataset, 
    num_workers = configs["num_workers"],
    batch_size = configs["batch_size"],
    shuffle = True,
    collate_fn = collate,
    drop_last = configs["drop_last"]
)

val_dataset = torch.load(Path(configs["data_dir"]) / "Validation" / "dataset.pt")
val_loader = DataLoader(
    dataset = val_dataset, 
    num_workers = configs["num_workers"],
    batch_size = configs["batch_size"],
    shuffle = True,
    collate_fn = collate,
    drop_last = configs["drop_last"]
)

criterion = multiloss
model = AnatomyModel(configs["input_size"], configs["output_size"])
optimizer = optim.AdamW(model.parameters(), lr = configs["lr"], weight_decay = configs["weight_decay"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode = 'min', 
    min_lr = 1e-6,      
    factor = 0.5,        
    patience = 5,         
    threshold = configs["es_thresh"]
)
trainer = Trainer(model, configs, train_loader, val_loader, criterion, optimizer, scheduler)
trainer.train()