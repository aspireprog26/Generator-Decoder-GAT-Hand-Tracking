from pathlib import Path

import earlystopper as es
import numpy as np
import optuna
import torch
from torch import device, nn, optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        configs: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim,
        scheduler: optim,
        criterion: nn,
        device: device,
    ):
        self.device = device
        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

        self.num_epochs = configs["num_epochs"]
        self.min_delta = configs["es_thresh"]
        patience = configs["es_patience"]

        decoder_stats = torch.load(
            Path(configs["stereo_data_dir"]) / "Training" / "Decoder" / "stats.pt",
            weights_only=False,
        )
        self.decoder_mean = decoder_stats[0].to(self.device, dtype=torch.float32)
        self.decoder_std = decoder_stats[1].to(self.device, dtype=torch.float32)

        self.model_save_path = (
            Path(configs["model_dir"]) / f"{configs['decoder_model_name']}reg"
        )
        self.best_loss = np.inf
        self.early_stopper = es.EarlyStopping(
            patience, self.min_delta, self.model_save_path
        )

    def standardize(self, features):
        stand_feats = (features - self.decoder_mean) / self.decoder_std
        return stand_feats

    def train(self, trial):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_dist = 0
            train_anatomy = 0

            for batch, target, coords_proj in self.train_loader:
                batch = batch.to(self.device)
                target = target.to(self.device)
                coords_proj = coords_proj.to(self.device)

                features = batch.x.to(self.device, dtype=torch.float32)
                features = features.reshape(batch.num_graphs, 21, 19)
                features = self.standardize(features)
                edge_index = batch.edge_index.to(self.device)
                b = batch.batch.to(self.device)

                self.optimizer.zero_grad()
                errors = self.model(features, edge_index, b)
                errors = errors.view(errors.size(0), 21, 3)
                scale = torch.linalg.norm(coords_proj[:, 9] - coords_proj[:, 0], dim=1)
                pred = coords_proj + (scale[:, None, None] * errors)
                loss, dist, anatomy_loss = self.criterion(pred, target)

                loss.backward()
                self.optimizer.step()
                train_dist += dist.item()
                train_anatomy += anatomy_loss.item()

            train_dist /= len(self.train_loader)
            train_anatomy /= len(self.train_loader)

            self.model.eval()
            val_dist = 0
            val_anatomy = 0

            with torch.inference_mode():
                for batch, target, coords_proj in self.val_loader:
                    batch = batch.to(self.device)
                    target = target.to(self.device)
                    coords_proj = coords_proj.to(self.device)

                    features = batch.x.to(self.device, dtype=torch.float32)
                    features = features.reshape(batch.num_graphs, 21, 19)
                    features = self.standardize(features)
                    edge_index = batch.edge_index.to(self.device)
                    b = batch.batch.to(self.device)

                    errors = self.model(features, edge_index, b)
                    errors = errors.view(errors.size(0), 21, 3)
                    scale = torch.linalg.norm(
                        coords_proj[:, 9] - coords_proj[:, 0], dim=1
                    )
                    pred = coords_proj + (scale[:, None, None] * errors)

                    _, dist, anatomy_loss = self.criterion(pred, target)
                    val_dist += dist.item()
                    val_anatomy += anatomy_loss.item()

            val_dist /= len(self.val_loader)
            val_anatomy /= len(self.val_loader)

            if self.scheduler is not None:
                self.scheduler.step(val_anatomy)

            gat_lr = self.optimizer.param_groups[0]["lr"]
            reg_lr = self.optimizer.param_groups[1]["lr"]

            print(
                f"Epoch: {epoch + 1} | "
                f"T-AL: {train_anatomy: .6f} | "
                f"T-D: {train_dist: .6f} | "
                f"V-AL: {val_anatomy: .6f} | "
                f"V-D: {val_dist: .6f} | "
                f"G-LR: {gat_lr: .6f} | "
                f"R-LR: {reg_lr: .6f}"
            )

            """
            if val_anatomy < self.best_loss - self.min_delta:
                self.best_loss = val_anatomy
                torch.save(self.model.state_dict(), self.model_save_path)

            if trial is not None:
                trial.report(val_anatomy, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            """

            self.early_stopper(val_anatomy, self.model)
            if self.early_stopper.stopping:
                print(f"Early Stopping at epoch {epoch + 1} / {self.num_epochs}")
                break

        return train_anatomy, val_anatomy
