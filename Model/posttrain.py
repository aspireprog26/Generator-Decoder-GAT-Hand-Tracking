import json
from pathlib import Path

import optuna
import torch
from model import AnatomyModel
from posttrainer import Trainer
from torch import nn, optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from utils import ANGLE_JOINTS, HAND_SKELETON, saveConfigs


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


MODE = "optuna"

with open("/home/miket/Documents/Hand-Tracking-2/Model/decpreconfigs.json", "r") as f:
    dec_post_configs = json.load(f)

"""
For post-optuna fine-tuning
with open("/home/miket/Documents/Hand-Tracking-2/Model/decpostconfigs.json", "r") as f:
    dec_post_configs = json.load(f)
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = Loss(
    dec_post_configs["delta1"],
    dec_post_configs["delta2"],
    0.5,
    0.125,
    0.125,
    0.125,
).criterion


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
        cfgs["decoder_hidden1"],
        cfgs["decoder_output_size"],
        cfgs["decoder_dropout"],
        mano_root=cfgs["mano_root"],
        generator=False,
        ncomps=cfgs["ncomps"],
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
        factor=0.5,
        patience=5,
        threshold=cfgs["es_thresh"],
    )

    trainer = Trainer(
        model, cfgs, train_loader, val_loader, optimizer, scheduler, criterion, device
    )
    try:
        train_loss, val_loss, train_dist, val_dist = trainer.train(trial)
    except optuna.TrialPruned:
        raise  # let Optuna's own pruning mechanism work normally
    except Exception as e:  # noqa: BLE001
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.TrialPruned()  # tell Optuna to treat this as a failed/pruned trial

    return train_loss, val_loss, train_dist, val_dist


def objective(trial):
    trial_configs = dec_post_configs.copy()
    num_epochs = trial.suggest_int("num_epochs", 30, 150, step=10)
    decoder_dropout = trial.suggest_float("decoder_dropout", 0, 0.6)
    lr_factor = trial.suggest_float("lr_factor", 0.05, 3)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    trial_configs.update(
        {
            "num_epochs": num_epochs,
            "decoder_dropout": decoder_dropout,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "lr_factor": lr_factor,
        }
    )

    train_loss, val_loss, train_dist, val_dist = train(trial_configs, criterion, trial)
    print(
        f"\nTrial Number: {trial.number} | "
        f"Train Loss: {train_loss: .4f} | "
        f"Train Anatomy: {train_dist: .4f} | "
        f"Val Loss: {val_loss: .4f} | "
        f"Val Anatomy: {val_dist:.4f}"
    )
    return val_loss


if __name__ == "__main__":
    if MODE == "optuna":
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
            storage="sqlite:///posttrainsearch.db",  # persists progress to disk
            study_name="posttrainsearch",
            load_if_exists=True,  # resume if the process restarts
        )
        study.optimize(objective, n_trials=100)

        print(f"Best loss: {study.best_value}")
        print("\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"{key}: {value}")

        dec_post_configs.update(study.best_params)
        saveConfigs(dec_post_configs, "decpostconfigs")
        train(dec_post_configs, criterion)
    else:
        train(dec_post_configs, criterion)
        saveConfigs(dec_post_configs, "decpostconfigs")
