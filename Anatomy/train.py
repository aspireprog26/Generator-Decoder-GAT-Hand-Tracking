import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from trainer import Trainer
from model import AnatomyModel
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

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

def unit_vec_loss(pred_vec, target_vec):
    return 1.0 - F.cosine_similarity(pred_vec, target_vec, dim = -1).mean()

def finger_block_loss(pred_f, target_f):
    # 0:3   CMC unit vector
    # 3:6   MCP unit vector
    # 6:9   IP unit vector
    # 9:12  TIP unit vector
    # 12:15 dot products
    # 15:18 sign values
    # 18:20 ratios

    vec_loss = (
        unit_vec_loss(pred_f[:, 0:3], target_f[:, 0:3]) +
        unit_vec_loss(pred_f[:, 3:6], target_f[:, 3:6]) +
        unit_vec_loss(pred_f[:, 6:9], target_f[:, 6:9]) +
        unit_vec_loss(pred_f[:, 9:12], target_f[:, 9:12])
    ) / 4.0

    dot_loss = F.smooth_l1_loss(pred_f[:, 12:15], target_f[:, 12:15])
    sign_loss = F.smooth_l1_loss(pred_f[:, 15:18], target_f[:, 15:18])
    ratio_loss = F.smooth_l1_loss(pred_f[:, 18:20], target_f[:, 18:20])

    total_loss = vec_loss + dot_loss + sign_loss + 0.5 * ratio_loss
    return total_loss

def multiloss(pred, target):
    pred_fingers = torch.chunk(pred, 5, dim=-1)
    target_fingers = torch.chunk(target, 5, dim=-1)

    losses = []
    for pf, tf in zip(pred_fingers, target_fingers):
        losses.append(finger_block_loss(pf, tf))

    return sum(losses) / len(losses)

def collate(batch):
    left_graphs, right_graphs, targets = zip(*batch)
    left_batch = Batch.from_data_list(list(left_graphs))
    right_batch = Batch.from_data_list(list(right_graphs))
    target = torch.stack(targets, dim = 0).float()
    return left_batch, right_batch, target

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