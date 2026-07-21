import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from posttrainer import Trainer
from model import AnatomyModel
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

with open("/Users/michaeltoppin/Documents/Coding/Hand-Tracking-2/Anatomy/configs.json", "r") as f:
    configs = json.load(f)

configs["lr"] = 1e-4
configs["dropout"] = 0.3
configs["batch_size"] = 32
configs["es_patience"] = 5
 
def collate(batch):
    graphs, targets, raw_coords = zip(*batch)
    batch = Batch.from_data_list(list(graphs))
    targets = torch.stack(targets, dim = 0).float()
    raw_coords = torch.stack(raw_coords, dim = 0).float()
    return (batch, targets, raw_coords)

train_dataset = torch.load(Path(configs["post_data_dir"]) / "Training" / "dataset.pt")
train_loader = DataLoader(
    dataset = train_dataset, 
    num_workers = configs["num_workers"],
    batch_size = configs["batch_size"],
    shuffle = True,
    drop_last = configs["drop_last"],
    collate_fn = collate
)

val_dataset = torch.load(Path(configs["post_data_dir"]) / "Validation" / "dataset.pt")
val_loader = DataLoader(
    dataset = val_dataset, 
    num_workers = configs["num_workers"],
    batch_size = configs["batch_size"],
    drop_last = configs["drop_last"],
    collate_fn = collate
)

criterion = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnatomyModel(configs["input_size"], configs["hidden_size"], configs["output_size"], configs["dropout"]).to(device)
weights = torch.load((Path(configs["model_dir"]) / configs["model_name"]), weights_only = True, map_location = device)
model.load_state_dict(weights)

optimizer = optim.AdamW(model.parameters(), lr = configs["lr"], weight_decay = configs["weight_decay"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode = 'min', 
    min_lr = 1e-6,      
    factor = configs["scheduler_factor"],        
    patience = configs["scheduler_patience"],         
    threshold = configs["es_thresh"]
)
trainer = Trainer(model, configs, train_loader, val_loader, criterion, optimizer, scheduler)
trainer.train()