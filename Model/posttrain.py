import json
import sys
from pathlib import Path

import optuna
import torch
from model import AnatomyModel
from posttrainer import Trainer
from pretrain import saveConfigs
from torch import nn, optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

sys.path.insert(0, "/home/miket/Documents/Hand-Tracking-2/Keypoints")
from keypointdetection import HAND_SKELETON, ANGLE_JOINTS  # type: ignore # noqa: I001


class Loss:
    def __init__(self, delta):
        self.bones = HAND_SKELETON
        self.angles = ANGLE_JOINTS
        self.delta = delta

        self.w1 = 0.4
        self.w2 = 0.6

    def distHuber(self, pred, target):
        dist = torch.linalg.norm(pred - target, dim=-1)
        quadratic = 0.5 * dist**2
        linear = self.delta * (dist - 0.5 * self.delta)
        loss = torch.where(dist < self.delta, quadratic, linear)
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
            loss += (length_target - length_pred) ** 2
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

    def handPointLoss(self, pred, target):
        weights = torch.ones(21, device=pred.device)
        weights[[1, 2, 3, 4]] = 5  # thumb
        weights[[5, 6, 7, 8]] = 2  # index
        weights[[9, 10, 11, 12]] = 5  # middle
        weights[[13, 14, 15, 16]] = 1  # ring
        weights[[17, 18, 19, 20]] = 1  # pinky

        diff = pred - target
        error = (diff**2).sum(dim=-1)
        weighted_error = (error * weights).sum(dim=-1)
        return (weighted_error / weights.sum()).mean()

    def criterion(self, pred, target):
        dist_loss, dist_mean = self.distHuber(pred, target)
        anatomy_loss = (
            self.boneDirLoss(pred, target)
            + self.boneLengthLoss(pred, target)
            + self.angleLoss(pred, target)
            + self.handPointLoss(pred, target)
        )
        loss = self.w1 * dist_loss + self.w2 * anatomy_loss
        return loss, dist_mean, anatomy_loss - self.handPointLoss(pred, target)


MODE = "optuna"
with open("/home/miket/Documents/Hand-Tracking-2/Model/decpreconfigs.json", "r") as f:
    dec_post_configs = json.load(f)

configs_pop = [
    "generator_lr",
    "generator_dropout",
    "generator_hidden_size",
    "generator_output_size",
    "generator_model_name",
    "stb_dir",
]

for config in configs_pop:
    dec_post_configs.pop(config)

"""
For fine tuning after trials
with open("/home/miket/Documents/Hand-Tracking-2/Model/decpostconfigs.json", "r") as f:
    dec_post_configs = json.load(f)
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate(batch):
    graphs, targets, raw_coords = zip(*batch)
    batch = Batch.from_data_list(list(graphs))
    targets = torch.stack(targets, dim=0).float()
    raw_coords = torch.stack(raw_coords, dim=0).float()
    return (batch, targets, raw_coords)


train_dataset = torch.load(
    Path(dec_post_configs["stereo_data_dir"]) / "Training" / "Decoder" / "dataset.pt",
    weights_only=False,
)

val_dataset = torch.load(
    Path(dec_post_configs["stereo_data_dir"]) / "Validation" / "Decoder" / "dataset.pt",
    weights_only=False,
)


def createDataset(batch_size):
    train_loader = DataLoader(
        dataset=train_dataset,
        num_workers=dec_post_configs["num_workers"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=dec_post_configs["drop_last"],
        collate_fn=collate,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        num_workers=dec_post_configs["num_workers"],
        batch_size=batch_size,
        drop_last=dec_post_configs["drop_last"],
        collate_fn=collate,
    )
    return train_loader, val_loader


def train(cfgs: dict, criterion, trial=None):
    train_loader, val_loader = createDataset(cfgs["batch_size"])

    model = AnatomyModel(
        cfgs["input_size"],
        cfgs["decoder_hidden_size"],
        cfgs["decoder_output_size"],
        cfgs["decoder_dropout"],
        generator=False,
    ).to(device)

    weights = torch.load(
        (Path(cfgs["model_dir"]) / f"pre{cfgs['decoder_model_name']}"),
        weights_only=True,
        map_location=device,
    )
    model.load_state_dict(weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfgs["decoder_lr"] * cfgs["lr_factor"],
        weight_decay=cfgs["weight_decay"],
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        min_lr=1e-6,
        factor=cfgs["scheduler_factor"],
        patience=cfgs["scheduler_patience"],
        threshold=cfgs["es_thresh"],
    )

    trainer = Trainer(
        model, cfgs, train_loader, val_loader, optimizer, scheduler, criterion, device
    )
    train_anatomy, val_anatomy = trainer.train(trial)
    return train_anatomy, val_anatomy


def objective(trial):
    trial_configs = dec_post_configs.copy()
    num_epochs = trial.suggest_int("num_epochs", 30, 150, step=5)
    decoder_dropout = trial.suggest_float("decoder_dropout", 0, 0.6)
    lr_factor = trial.suggest_float("lr_factor", 0.05, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    delta = trial.suggest_float("delta", 0.5, 10, log=True)

    criterion = Loss(delta).criterion
    trial_configs.update(
        {
            "num_epochs": num_epochs,
            "decoder_dropout": decoder_dropout,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "lr_factor": lr_factor,
            "delta": delta,
        }
    )

    train_anatomy, val_anatomy = train(trial_configs, criterion, trial)
    print(
        f"\nTrial Number: {trial.number} | "
        f"Train Loss: {train_anatomy} | "
        f"Validation Loss: {val_anatomy}"
    )
    return val_anatomy


if __name__ == "__main__":
    if MODE == "optuna":
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=15),
        )
        study.optimize(objective, n_trials=50)

        print(f"Best loss: {study.best_value}")
        print("\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"{key}: {value}")

        dec_post_configs.update(study.best_params)
        saveConfigs(dec_post_configs, "decpostconfigs")
        final_criterion = Loss(dec_post_configs["delta"]).criterion
        train(dec_post_configs, final_criterion)
    else:
        final_criterion = Loss(dec_post_configs["delta"]).criterion
        train(dec_post_configs, final_criterion)
        saveConfigs(dec_post_configs, "decpostconfigs")
