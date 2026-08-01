from pathlib import Path

# import earlystopper as es
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
        # patience = configs["es_patience"]

        stats = torch.load(
            Path(configs["stereo_data_dir"]) / "Training" / "Decoder" / "stats.pt",
            weights_only=False,
        )
        self.mean = stats[0].to(self.device, dtype=torch.float32)
        self.std = stats[1].to(self.device, dtype=torch.float32)

        self.model_save_path = Path(configs["model_dir"]) / f"{configs['model_name']}"
        self.best_loss = np.inf
        # self.early_stopper = es.EarlyStopping(patience, self.min_delta, model_save_path)

    def standardize(self, features):
        stand_feats = (features - self.mean) / self.std
        return stand_feats

    def train(self, trial):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_dist = 0
            train_loss = 0
            train_samples = 0

            for batch, target, _ in self.train_loader:
                batch = batch.to(self.device)
                target = target.to(self.device)

                features = batch.x.to(self.device, dtype=torch.float32)
                features = features.reshape(batch.num_graphs, 21, 19)
                features = self.standardize(features)
                edge_index = batch.edge_index.to(self.device)
                b = batch.batch.to(self.device)

                self.optimizer.zero_grad()
                pred_coords = self.model(features, edge_index, b)

                pred_scale = torch.linalg.norm(
                    pred_coords[:, 9] - pred_coords[:, 0], dim=-1, keepdim=True
                ).clamp_min(1e-8)
                pred_coords = pred_coords - pred_coords[:, :1]
                pred_coords_norm = pred_coords / pred_scale.unsqueeze(-1)

                target_scale = torch.linalg.norm(
                    target[:, 9] - target[:, 0], dim=-1, keepdim=True
                ).clamp_min(1e-8)
                target = target - target[:, :1]
                target_norm = target / target_scale.unsqueeze(-1)

                loss, dist = self.criterion(pred_coords_norm, target_norm)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_dist += dist.item() * batch.num_graphs
                train_loss += loss.item() * batch.num_graphs
                train_samples += batch.num_graphs

            train_dist /= train_samples
            train_loss /= train_samples

            self.model.eval()
            val_dist = 0
            val_loss = 0
            val_samples = 0

            with torch.inference_mode():
                for batch, target, _ in self.val_loader:
                    batch = batch.to(self.device)
                    target = target.to(self.device)

                    features = batch.x.to(self.device, dtype=torch.float32)
                    features = features.reshape(batch.num_graphs, 21, 19)
                    features = self.standardize(features)
                    edge_index = batch.edge_index.to(self.device)
                    b = batch.batch.to(self.device)

                    pred_coords = self.model(features, edge_index, b)

                    pred_scale = torch.linalg.norm(
                        pred_coords[:, 9] - pred_coords[:, 0], dim=-1, keepdim=True
                    ).clamp_min(1e-8)
                    pred_coords = pred_coords - pred_coords[:, :1]
                    pred_coords_norm = pred_coords / pred_scale.unsqueeze(-1)

                    target_scale = torch.linalg.norm(
                        target[:, 9] - target[:, 0], dim=-1, keepdim=True
                    ).clamp_min(1e-8)
                    target = target - target[:, :1]
                    target_norm = target / target_scale.unsqueeze(-1)

                    loss, dist = self.criterion(pred_coords_norm, target_norm)

                    val_dist += dist.item() * batch.num_graphs
                    val_loss += loss.item() * batch.num_graphs
                    val_samples += batch.num_graphs

            val_dist /= val_samples
            val_loss /= val_samples

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch: {epoch + 1} | "
                f"T-AL: {train_loss: .6f} | "
                f"T-D: {train_dist: .6f} | "
                f"V-AL: {val_loss: .6f} | "
                f"V-D: {val_dist: .6f} | "
                f"LR: {lr: .6f}"
            )

            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), self.model_save_path)

            if trial is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            """
            self.early_stopper(val_loss, self.model)
            if self.early_stopper.stopping:
                print(f"Early Stopping at epoch {epoch + 1} / {self.num_epochs}")
                break
            """

        return (
            train_loss,
            val_loss,
            train_dist,
            val_dist,
        )
