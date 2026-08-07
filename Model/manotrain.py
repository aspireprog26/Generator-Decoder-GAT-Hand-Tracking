import json
from pathlib import Path

import optuna
import torch
from manotrainer import Trainer
from model import AnatomyModel, MANOModel
from torch import nn, optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from utils import ANGLE_JOINTS, HAND_SKELETON, saveConfigs

MODE = "optuna"

with open("/home/miket/Documents/Hand-Tracking-2/Model/decpostconfigs.json", "r") as f:
    dec_post_configs = json.load(f)

configs = {
    "model_name": "model.pth",
    "mano_root": "/home/miket/Documents/mano/models",
    "model_dir": "/home/miket/Documents/Hand-Tracking-2/Model",
    "stereo_data_dir": "/home/miket/Documents/StereoDataset",
    "num_workers": 12,
    "weight_decay": 1e-2,
    "dropout": 0.1,
    "lr": 1e-3,
    "batch_size": 64,
    "input_size": 19,
    "ncomps": 45,
    "num_epochs": 100,
    "hidden_size": 32,
    "hidden1": 480,
    "es_patience": 10,
    "es_thresh": 1e-4,
    "scheduler": 0.5,
    "scheduler_patience": 5,
    "drop_last": False,
    "w1": 0.5,
    "w2": (0.5 / 3),
    "w3": (0.5 / 3),
    "w4": (0.5 / 3),
}

configs.update(
    {"delta1": dec_post_configs["delta1"], "delta2": dec_post_configs["delta2"]}
)

# For post Optuna fine tuning
"""
with open("/home/miket/Documents/Hand-Tracking-2/Model/manoconfigs.json", "r") as f:
    configs = json.load(f)
"""


class Loss:
    def __init__(self, delta1, delta2, w1, w2, w3, w4):
        self.bones = HAND_SKELETON
        self.angles = ANGLE_JOINTS
        self.delta1 = delta1
        self.huber = nn.HuberLoss(delta=delta2, reduction="none")

        total = w1 + w2 + w3 + w4
        self.w1 = w1 / total
        self.w2 = w2 / total
        self.w3 = w3 / total
        self.w4 = w4 / total

    def distHuber(self, pred, target):
        dist = torch.linalg.norm(pred - target, dim=-1)
        quadratic = 0.5 * dist**2
        linear = self.delta1 * (dist - 0.5 * self.delta1)
        loss = torch.where(dist < self.delta1, quadratic, linear)
        return loss.mean(), dist.mean()

    def boneDirLoss(self, pred, target):
        loss = 0
        for parent, child in self.bones:
            bone_pred = pred[:, child] - pred[:, parent]
            bone_target = target[:, child] - target[:, parent]

            bone_pred = nn.functional.normalize(bone_pred, dim=-1)
            bone_target = nn.functional.normalize(bone_target, dim=-1)

            loss += ((bone_pred - bone_target) ** 2).sum(dim=-1)
        return (loss / len(self.bones)).mean()

    def boneLengthLoss(self, pred, target):
        loss = 0
        for parent, child in self.bones:
            length_pred = torch.linalg.norm(pred[:, child] - pred[:, parent], dim=-1)
            length_target = torch.linalg.norm(
                target[:, child] - target[:, parent], dim=-1
            )
            loss += self.huber(length_target, length_pred)
        return (loss / len(self.bones)).mean()

    def angleLoss(self, pred, target):
        loss = 0
        for parent, joint, child in self.angles:
            p1 = pred[:, parent] - pred[:, joint]
            p2 = pred[:, child] - pred[:, joint]

            t1 = target[:, parent] - target[:, joint]
            t2 = target[:, child] - target[:, joint]

            cos_pred = nn.functional.cosine_similarity(p1, p2, dim=-1)
            cos_target = nn.functional.cosine_similarity(t1, t2, dim=-1)
            loss += (cos_pred - cos_target) ** 2
        return (loss / len(self.angles)).mean()

    def criterion(self, pred, target):
        dist_loss, dist_mean = self.distHuber(pred, target)
        bone_dir_loss = self.boneDirLoss(pred, target)
        bone_length_loss = self.boneLengthLoss(pred, target)
        angle_loss = self.angleLoss(pred, target)

        loss = (
            self.w1 * dist_loss
            + self.w2 * bone_dir_loss
            + self.w3 * bone_length_loss
            + self.w4 * angle_loss
        )
        return loss, dist_mean


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = Loss(
    configs["delta1"],
    configs["delta2"],
    configs["w1"],
    configs["w2"],
    configs["w3"],
    configs["w4"],
).criterion


def collate(batch):
    graphs, targets, raw_coords = zip(*batch)
    batch = Batch.from_data_list(list(graphs))
    targets = torch.stack(targets, dim=0).float()
    raw_coords = torch.stack(raw_coords, dim=0).float()
    return (batch, targets, raw_coords)


train_dataset = torch.load(
    Path(configs["stereo_data_dir"]) / "Training" / "Decoder" / "dataset.pt",
    weights_only=False,
)

val_dataset = torch.load(
    Path(configs["stereo_data_dir"]) / "Validation" / "Decoder" / "dataset.pt",
    weights_only=False,
)


def createDataset(batch_size):
    train_loader = DataLoader(
        dataset=train_dataset,
        num_workers=configs["num_workers"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=configs["drop_last"],
        collate_fn=collate,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        num_workers=configs["num_workers"],
        batch_size=batch_size,
        drop_last=configs["drop_last"],
        collate_fn=collate,
    )
    return train_loader, val_loader


def train(cfgs: dict, criterion, trial=None):
    train_loader, val_loader = createDataset(cfgs["batch_size"])

    joint_predictor = AnatomyModel(
        dec_post_configs["input_size"],
        dec_post_configs["decoder_hidden_size"],
        dec_post_configs["decoder_hidden1"],
        dec_post_configs["decoder_output_size"],
        dec_post_configs["decoder_dropout"],
        generator=False,
    ).to(device)

    weights = torch.load(
        (
            Path(dec_post_configs["model_dir"])
            / f"{dec_post_configs['decoder_model_name']}"
        ),
        weights_only=True,
        map_location=device,
    )
    joint_predictor.load_state_dict(weights)

    # Freeze the joint predictor model
    for param in joint_predictor.parameters():
        param.requires_grad = False

    mano_model = MANOModel(
        cfgs["input_size"],
        cfgs["hidden_size"],
        cfgs["hidden1"],
        cfgs["dropout"],
        cfgs["mano_root"],
        cfgs["ncomps"],
    ).to(device)

    optimizer = optim.AdamW(
        mano_model.parameters(),
        lr=cfgs["lr"],
        weight_decay=cfgs["weight_decay"],
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        min_lr=1e-6,
        factor=0.5,
        patience=5,
        threshold=cfgs["es_thresh"],
    )

    trainer = Trainer(
        mano_model,
        joint_predictor,
        cfgs,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        device,
    )
    try:
        train_loss, val_loss, train_dist, val_dist = trainer.train(trial)
    except optuna.TrialPruned:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.TrialPruned()

    return train_loss, val_loss, train_dist, val_dist


def objective(trial):
    trial_configs = configs.copy()
    num_epochs = trial.suggest_int("num_epochs", 30, 150, step=10)
    dropout = trial.suggest_float("dropout", 0, 0.6)
    ncomps = trial.suggest_int("ncomps", 6, 45)
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_int("hidden_size", 32, 128, step=16)
    hidden1 = trial.suggest_int("hidden1", 128, 512, step=16)

    trial_configs.update(
        {
            "num_epochs": num_epochs,
            "hidden_size": hidden_size,
            "hidden1": hidden1,
            "dropout": dropout,
            "ncomps": ncomps,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "lr": lr,
        }
    )

    train_loss, val_loss, train_dist, val_dist = train(trial_configs, criterion, trial)
    print(
        f"\nTrial Number: {trial.number} | "
        f"Train Loss: {train_loss: .4f} | "
        f"Train VDL: {train_dist: .4f} | "
        f"Val Loss: {val_loss: .4f} | "
        f"Val VDL: {val_dist:.4f}"
    )
    return val_loss


if __name__ == "__main__":
    if MODE == "optuna":
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
            storage="sqlite:///manotrainsearch.db",
            study_name="manotrainsearch",
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=100)

        print(f"Best loss: {study.best_value}")
        print("\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"{key}: {value}")

        configs.update(study.best_params)
        saveConfigs(configs, "manoconfigs")
        train(configs, criterion)
    else:
        train(configs, criterion)
        saveConfigs(configs, "manoconfigs")
