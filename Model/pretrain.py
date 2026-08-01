import json
import random
from pathlib import Path

import numpy as np
import optuna
import torch
from model import AnatomyModel
from pretrainer import Trainer
from torch import nn, optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from utils import ANGLE_JOINTS, HAND_SKELETON, saveConfigs

# Must use for reproducability
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

g = torch.Generator()
g.manual_seed(SEED)

MODE = "optuna"
configs = {
    "decoder_lr": 1e-3,
    "generator_lr": 1e-3,
    "generator_dropout": 0.2,
    "decoder_dropout": 0.2,
    "batch_size": 64,
    "input_size": 19,
    "decoder_hidden_size": 32,
    "generator_hidden1": 128,
    "decoder_hidden1": 64,
    "ncomps": 6,
    "generator_output_size": 64,
    "decoder_output_size": 22,  # ncomps + 10 + 3 + 3
    "num_workers": 2,
    "num_epochs": 100,
    "delta": 1,
    "weight_decay": 1e-2,
    "decoder_model_name": "decoder.pth",
    "generator_model_name": "generator.pth",
    "model_dir": "/home/miket/Documents/Hand-Tracking-2/Model",
    "mano_root": "/home/miket/Documents/mano/models",
    "stb_dir": "/home/miket/Documents/StereoSTBDataset",
    "stereo_data_dir": "/home/miket/Documents/StereoDataset",
    "es_patience": 10,
    "es_thresh": 1e-4,
    "scheduler_factor": 0.5,
    "scheduler_patience": 5,
    "drop_last": False,
}

# For fine tuning after optuna trials are complete, MODE="train"
with open("/home/miket/Documents/Hand-Tracking-2/Model/optunaconfigs.json", "r") as f:
    dec_pre_configs = json.load(f)

log2pi = torch.log(torch.tensor(2 * torch.pi))


class Loss:
    def __init__(self, delta1, delta2, w1, w2, w3, w4):
        self.bones = HAND_SKELETON
        self.angles = ANGLE_JOINTS
        self.delta1 = delta1
        self.huber = nn.HuberLoss(delta=delta2, reduction="none")

        # Normalize so weights sum to one
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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collateDecoder(batch):
    graphs, left_kps, right_kps = zip(*batch)
    batch = Batch.from_data_list(list(graphs))
    left_kps = torch.stack(left_kps, dim=0).float()
    right_kps = torch.stack(right_kps, dim=0).float()
    return (batch, left_kps, right_kps)


def collateGenerator(batch):
    graphs, targets, raw_coords = zip(*batch)
    batch = Batch.from_data_list(list(graphs))
    targets = torch.stack(targets, dim=0).float()
    raw_coords = torch.stack(raw_coords, dim=0).float()
    return (batch, targets, raw_coords)


def generatorCriterion(X, M, scale, U, Vinv, logdet_V, device):
    dtype = X.dtype

    X = X.to(device=device, dtype=dtype)
    M = M.to(device=device, dtype=dtype)
    scale = scale.to(device=device, dtype=dtype)
    U = U.to(device=device, dtype=dtype)
    Vinv = Vinv.to(device=device, dtype=dtype)
    logdet_V = torch.as_tensor(logdet_V, device=device, dtype=dtype)
    log2pi_ = log2pi.to(device=device, dtype=dtype)

    _, m, n = X.shape
    cov_row = scale[:, None, None] * U
    E = X - M

    logdet_U = torch.linalg.slogdet(cov_row).logabsdet
    Uinv = torch.linalg.inv(cov_row)

    quad = torch.einsum("bij,bjk,kl,bli->b", Uinv, E, Vinv, E.transpose(-1, -2))
    nll = 0.5 * (m * n * log2pi_ + n * logdet_U + m * logdet_V + quad)
    return nll.mean()


generator_criterion = generatorCriterion

decoder_train_dataset = torch.load(
    Path(configs["stb_dir"]) / "Training" / "dataset.pt", weights_only=False
)
decoder_val_dataset = torch.load(
    Path(configs["stb_dir"]) / "Validation" / "dataset.pt", weights_only=False
)

generator_train_dataset = torch.load(
    Path(configs["stereo_data_dir"]) / "Training" / "Generator" / "dataset.pt",
    weights_only=False,
)
generator_val_dataset = torch.load(
    Path(configs["stereo_data_dir"]) / "Validation" / "Generator" / "dataset.pt",
    weights_only=False,
)


def createDataset(batch_size):
    decoder_train_loader = DataLoader(
        dataset=decoder_train_dataset,
        num_workers=configs["num_workers"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collateDecoder,
        drop_last=configs["drop_last"],
        worker_init_fn=seed_worker,
        generator=g,
    )

    decoder_val_loader = DataLoader(
        dataset=decoder_val_dataset,
        num_workers=configs["num_workers"],
        batch_size=batch_size,
        collate_fn=collateDecoder,
        drop_last=configs["drop_last"],
        worker_init_fn=seed_worker,
    )

    generator_train_loader = DataLoader(
        dataset=generator_train_dataset,
        num_workers=configs["num_workers"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collateGenerator,
        drop_last=configs["drop_last"],
        worker_init_fn=seed_worker,
        generator=g,
    )

    generator_val_loader = DataLoader(
        dataset=generator_val_dataset,
        num_workers=configs["num_workers"],
        batch_size=batch_size,
        collate_fn=collateGenerator,
        drop_last=configs["drop_last"],
        worker_init_fn=seed_worker,
    )

    return (
        decoder_train_loader,
        decoder_val_loader,
        generator_train_loader,
        generator_val_loader,
    )


def train(cfgs: dict, decoder_criterion, trial=None):
    (
        decoder_train_loader,
        decoder_val_loader,
        generator_train_loader,
        generator_val_loader,
    ) = createDataset(cfgs["batch_size"])

    decoder_model = AnatomyModel(
        cfgs["input_size"],
        cfgs["decoder_hidden_size"],
        cfgs["decoder_hidden1"],
        cfgs["decoder_output_size"],
        cfgs["decoder_dropout"],
        mano_root=cfgs["mano_root"],
        generator=False,
        ncomps=cfgs["ncomps"],
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
        factor=0.5,
        patience=5,
        threshold=cfgs["es_thresh"],
    )

    generator_model = AnatomyModel(
        cfgs["input_size"],
        cfgs["generator_hidden_size"],
        cfgs["generator_hidden1"],
        cfgs["generator_output_size"],
        cfgs["generator_dropout"],
        mano_root=cfgs["mano_root"],
        generator=True,
        ncomps=cfgs["ncomps"],
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
        factor=0.5,
        patience=5,
        threshold=cfgs["es_thresh"],
    )

    trainer = Trainer(
        cfgs,
        decoder_criterion,
        generator_criterion,
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
    train_loss, val_loss, train_dist, val_dist = trainer.train(trial)
    return train_loss, val_loss, train_dist, val_dist


def objective(trial):
    trial_configs = configs.copy()
    decoder_hidden1 = trial.suggest_int("decoder_hidden1", 128, 512, step=16)
    generator_hidden1 = trial.suggest_int("generator_hidden1", 128, 512, step=16)
    generator_hidden_size = trial.suggest_int("generator_hidden_size", 32, 128, step=16)
    decoder_hidden_size = trial.suggest_int("decoder_hidden_size", 32, 128, step=16)
    generator_dropout = trial.suggest_float("generator_dropout", 0.0, 0.4)
    decoder_dropout = trial.suggest_float("decoder_dropout", 0.0, 0.6)
    delta1 = trial.suggest_float("delta1", 1, 10, log=True)
    delta2 = trial.suggest_float("delta2", 1, 10, log=True)
    decoder_lr = trial.suggest_float("decoder_lr", 5e-5, 1e-3, log=True)
    generator_lr = trial.suggest_float("generator_lr", 5e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    w1 = trial.suggest_float("w1", 0, 1)
    w2 = trial.suggest_float("w2", 0, 1)
    w3 = trial.suggest_float("w3", 0, 1)
    w4 = trial.suggest_float("w4", 0, 1)
    num_epochs = trial.suggest_int("num_epochs", 30, 80, step=10)

    trial_configs.update(
        {
            "decoder_lr": decoder_lr,
            "generator_lr": generator_lr,
            "decoder_hidden1": decoder_hidden1,
            "generator_hidden1": generator_hidden1,
            "generator_hidden_size": generator_hidden_size,
            "decoder_hidden_size": decoder_hidden_size,
            "generator_dropout": generator_dropout,
            "decoder_dropout": decoder_dropout,
            "w1": w1,
            "w2": w2,
            "w3": w3,
            "w4": w4,
            "delta1": delta1,
            "delta2": delta2,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
        }
    )
    decoder_criterion = Loss(delta1, delta2, w1, w2, w3, w4).criterion

    train_loss, val_loss, train_dist, val_dist = train(
        trial_configs, decoder_criterion, trial
    )
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
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=15),
        )
        study.optimize(objective, n_trials=65)

        print(f"Best loss: {study.best_value}")
        print("\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"{key}: {value}")

        configs.update(study.best_params)
        saveConfigs(configs, "optunaconfigs")
        final_decoder_criterion = Loss(
            configs["delta1"],
            configs["delta2"],
            configs["w1"],
            configs["w2"],
            configs["w3"],
            configs["w4"],
        ).criterion
        train(configs, final_decoder_criterion)
    else:
        final_decoder_criterion = Loss(
            dec_pre_configs["delta1"],
            dec_pre_configs["delta2"],
            dec_pre_configs["w1"],
            dec_pre_configs["w2"],
            dec_pre_configs["w3"],
            dec_pre_configs["w4"],
        ).criterion
        train(dec_pre_configs, final_decoder_criterion)
        saveConfigs(dec_pre_configs, "decpreconfigs")
