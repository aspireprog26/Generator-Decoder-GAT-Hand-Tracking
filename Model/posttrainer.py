from pathlib import Path

import earlystopper as es
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        configs: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn,
        optimizer: optim,
        scheduler,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.num_epochs = configs["num_epochs"]
        min_delta = configs["es_thresh"]
        patience = configs["es_patience"]

        decoder_stats = torch.load(
            Path(configs["stereo_data_dir"]) / "Training" / "Decoder" / "stats.pt",
            weights_only=False,
        )
        self.decoder_mean = decoder_stats[0].to(self.device, dtype=torch.float32)
        self.decoder_std = decoder_stats[1].to(self.device, dtype=torch.float32)

        model_save_path = Path(configs["model_dir"]) / configs["model_name"]
        self.early_stopper = es.EarlyStopping(patience, min_delta, model_save_path)

    def standardize(self, features):
        stand_feats = (features - self.decoder_mean) / self.decoder_std
        return stand_feats

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0

            for batch, target, coords_proj in self.train_loader:
                batch = batch.to(self.device)
                target = target.to(self.device)
                coords_proj = coords_proj.to(self.device)

                features = batch.x
                features = self.standardize(features)
                edge_index = batch.edge_index
                b = batch.batch

                self.optimizer.zero_grad()
                error = self.model(features, edge_index, b)
                scale = torch.linalg.norm(coords_proj[:, 9] - coords_proj[:, 0], dim=1)
                pred = coords_proj + (scale[:, None, None] * error)
                loss = self.criterion(pred, target)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)

            self.model.eval()
            val_loss = 0

            with torch.inference_mode():
                for batch, target, coords_proj in self.val_loader:
                    batch = batch.to(self.device)
                    target = target.to(self.device)
                    coords_proj = coords_proj.to(self.device)

                    features = batch.x
                    features = self.standardize(features)
                    edge_index = batch.edge_index
                    b = batch.batch

                    error = self.model(features, edge_index, b)
                    scale = torch.linalg.norm(
                        coords_proj[:, 9] - coords_proj[:, 0], dim=1
                    )
                    pred = coords_proj + (scale[:, None, None] * error)

                    loss = self.criterion(pred, target)
                    val_loss += loss.item()
            val_loss /= len(self.val_loader)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch: {epoch + 1} | Train Loss: {train_loss} | Val Loss: {val_loss} | LR: {current_lr}"
            )

            self.early_stopper(val_loss, self.model)
            if self.early_stopper.stopping:
                print(f"Early Stopping at epoch {epoch + 1} / {self.num_epochs}")
                break
