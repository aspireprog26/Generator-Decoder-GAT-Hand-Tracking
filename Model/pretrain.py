import torch
import json
import optuna
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from model import AnatomyModel
from pretrainer import Trainer
from torch_geometric.data import Batch
from torch.utils.data import DataLoader

configs = {
    "decoder_lr": 1e-3,
    "generator_lr": 1e-3,
    "generator_dropout": 0.2,
    "decoder_dropout": 0.2,
    "batch_size": 64,
    "input_size": 7,
    "decoder_hidden_size": 32,
    "generator_hidden_size": 64,
    "decoder_output_size": 3,
    "generator_output_size": 65,
    "num_workers": 2,
    "num_epochs": 250,
    "weight_decay": 1e-2,
    "decoder_model_name": "decoder.pth",
    "generator_model_name": "generator.pth",
    "model_dir": "/home/mrtcloud-1/Documents/Hand-Tracking-2/Model",
    "data_dir": "/home/mrtcloud-1/Documents/StereoSTBDataset",
    "post_data_dir": "/home/mrtcloud-1/Documents/StereoDataset",
    "es_patience": 10,
    "es_thresh": 1e-4,
    "scheduler_factor": 0.5,
    "scheduler_patience": 5,
    "drop_last": False,
}


def saveConfigs():
    with open((Path(configs["model_dir"]) / "configs.json"), "w") as f:
        json.dump(configs, f, indent=4)


def collateDecoder(batch):
    coords, _ = zip(*batch)
    coord_batch = Batch.from_data_list(list(coords))
    return coord_batch


def collateGenerator(batch):
    graphs, targets, raw_coords = zip(*batch)
    batch = Batch.from_data_list(list(graphs))
    targets = torch.stack(targets, dim=0).float()
    raw_coords = torch.stack(raw_coords, dim=0).float()
    return (batch, targets, raw_coords)


criterion = nn.MSELoss()
decoder_train_dataset = torch.load(
    Path(configs["data_dir"]) / "Training" / "dataset.pt"
)
decoder_val_dataset = torch.load(
    Path(configs["data_dir"]) / "Validation" / "dataset.pt"
)

generator_train_dataset = torch.load(
    Path(configs["post_data_dir"]) / "Training" / "dataset.pt"
)
generator_val_dataset = torch.load(
    Path(configs["post_data_dir"]) / "Validation" / "dataset.pt"
)


def createDataset(batch_size):
    decoder_train_loader = DataLoader(
        dataset=decoder_train_dataset,
        num_workers=configs["num_workers"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collateDecoder,
        drop_last=configs["drop_last"],
    )

    decoder_val_loader = DataLoader(
        dataset=decoder_val_dataset,
        num_workers=configs["num_workers"],
        batch_size=batch_size,
        collate_fn=collateDecoder,
        drop_last=configs["drop_last"],
    )

    generator_train_loader = DataLoader(
        dataset=generator_train_dataset,
        num_workers=configs["num_workers"],
        batch_size=batch_size * 3,
        shuffle=True,
        collate_fn=collateGenerator,
        drop_last=configs["drop_last"],
    )

    generator_val_loader = DataLoader(
        dataset=generator_val_dataset,
        num_workers=configs["num_workers"],
        batch_size=batch_size * 3,
        collate_fn=collateGenerator,
        drop_last=configs["drop_last"],
    )

    return (
        decoder_train_loader,
        decoder_val_loader,
        generator_train_loader,
        generator_val_loader,
    )


def train(cfgs: dict, trial=None):
    (
        decoder_train_loader,
        decoder_val_loader,
        generator_train_loader,
        generator_val_loader,
    ) = createDataset(cfgs["batch_size"])

    decoder_model = AnatomyModel(
        cfgs["input_size"],
        cfgs["decoder_hidden_size"],
        cfgs["decoder_output_size"],
        cfgs["decoder_dropout"],
        generator=False,
    )

    decoder_optimizer = optim.AdamW(
        decoder_model.parameters(),
        lr=cfgs["decoder_lr"],
        weight_decay=cfgs["weight_decay"],
    )

    decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        decoder_optimizer,
        mode="min",
        min_lr=1e-6,
        factor=cfgs["scheduler_factor"],
        patience=cfgs["scheduler_patience"],
        threshold=cfgs["es_thresh"],
    )

    generator_model = AnatomyModel(
        cfgs["input_size"],
        cfgs["generator_hidden_size"],
        cfgs["generator_output_size"],
        cfgs["generator_dropout"],
        generator=True,
    )

    generator_optimizer = optim.AdamW(
        generator_model.parameters(),
        lr=cfgs["generator_lr"],
        weight_decay=cfgs["weight_decay"],
    )

    generator_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        generator_optimizer,
        mode="min",
        min_lr=1e-6,
        factor=cfgs["scheduler_factor"],
        patience=cfgs["scheduler_patience"],
        threshold=cfgs["es_thresh"],
    )

    trainer = Trainer(
        cfgs,
        criterion,
        decoder_model,
        generator_model,
        decoder_train_loader,
        decoder_val_loader,
        generator_train_loader,
        generator_val_loader,
        decoder_optimizer,
        generator_optimizer,
        decoder_scheduler,
        generator_scheduler,
    )
    val_loss = trainer.train(trial)
    return val_loss


def objective(trial):
    trial_configs = configs.copy()
    generator_hidden_size = trial.suggest_int("generator_hidden_size", 32, 64, step=16)
    decoder_hidden_size = trial.suggest_int("decoder_hidden_size", 32, 64, step=16)
    generator_dropout = trial.suggest_float("generator_dropout", 0.1, 0.3)
    decoder_dropout = trial.suggest_float("decoder_dropout", 0.2, 0.4)
    decoder_lr = trial.suggest_float("decoder_lr", 1e-5, 1e-3, log=True)
    generator_lr = trial.suggest_float("generator_lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    scheduler_factor = trial.suggest_float("scheduler_factor", 0.2, 0.7)
    scheduler_patience = trial.suggest_int("scheduler_patience", 5, 10)

    trial_configs.update(
        {
            "decoder_lr": decoder_lr,
            "generator_lr": generator_lr,
            "generator_hidden_size": generator_hidden_size,
            "decoder_hidden_size": decoder_hidden_size,
            "generator_dropout": generator_dropout,
            "decoder_dropout": decoder_dropout,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "scheduler_factor": scheduler_factor,
            "scheduler_patience": scheduler_patience,
        }
    )

    train_loss, val_loss = train(trial_configs, trial)
    print(
        f"\nTrial Number: {trial.number} | Train Loss: {train_loss} | Validation Loss: {val_loss}"
    )
    return val_loss


study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
)
study.optimize(objective, n_trials=25)

print(f"Best loss: {study.best_value}")
print("\nBest parameters:")
for key, value in study.best_params.items():
    print(key, value)

configs.update(study.best_params)
saveConfigs()
train(configs, trial=None)
