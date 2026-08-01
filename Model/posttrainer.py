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

        decoder_stats = torch.load(
            Path(configs["stereo_data_dir"]) / "Training" / "Decoder" / "stats.pt",
            weights_only=False,
        )
        self.decoder_mean = decoder_stats[0].to(self.device, dtype=torch.float32)
        self.decoder_std = decoder_stats[1].to(self.device, dtype=torch.float32)

        self.model_save_path = (
            Path(configs["model_dir"]) / f"{configs['decoder_model_name']}"
        )
        self.best_loss = np.inf
        # self.early_stopper = es.EarlyStopping(patience, self.min_delta, model_save_path)

    def standardize(self, features):
        stand_feats = (features - self.decoder_mean) / self.decoder_std
        return stand_feats

    def procrustesAlign(
        self,
        X,
        Y,
        allow_reflection=False,
        allow_scaling=True,
        eps=1e-7,
    ):
        in_dtype = X.dtype

        # Use float64 for numerical stability
        X64 = X.double()
        Y64 = Y.double()

        B, N, D = X64.shape

        # Center point clouds
        X_mean = X64.mean(dim=1, keepdim=True)
        Y_mean = Y64.mean(dim=1, keepdim=True)

        X_c = X64 - X_mean
        Y_c = Y64 - Y_mean

        # Cross covariance
        M = torch.bmm(X_c.transpose(1, 2), Y_c)

        # SVD
        U, S, Vh = torch.linalg.svd(M)

        # Rotation
        if allow_reflection:
            R = torch.bmm(U, Vh)
            scale_num = S.sum(dim=-1)
        else:
            det = torch.linalg.det(torch.bmm(U, Vh))

            sign = torch.where(
                det < 0,
                -torch.ones_like(det),
                torch.ones_like(det),
            ).detach()

            Dmat = (
                torch.eye(
                    D,
                    device=X.device,
                    dtype=torch.float64,
                )
                .unsqueeze(0)
                .repeat(B, 1, 1)
            )

            Dmat[:, -1, -1] = sign
            R = torch.bmm(
                torch.bmm(U, Dmat),
                Vh,
            )
            scale_num = S.sum(dim=-1) - (sign < 0).to(S.dtype) * 2.0 * S[:, -1]

        # Scale
        if allow_scaling:
            var_X = (X_c**2).sum(dim=(1, 2))
            var_X = torch.clamp(var_X, min=eps)
            s = (scale_num / var_X)[:, None, None]
        else:
            s = torch.ones(
                (B, 1, 1),
                device=X.device,
                dtype=torch.float64,
            )

        # Apply alignment
        X_aligned = s * torch.bmm(X_c, R) + Y_mean
        return X_aligned.to(in_dtype)

    def train(self, trial):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_dist = 0
            train_anatomy = 0
            train_samples = 0

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
                pred_coords = self.model(features, edge_index, b)
                pred_coords = self.procrustesAlign(pred_coords, coords_proj)
                decoder_loss, dist = self.criterion(pred_coords, target)

                decoder_loss.backward()
                self.optimizer.step()

                train_dist += dist.item() * batch.num_graphs
                train_anatomy += decoder_loss.item() * batch.num_graphs
                train_samples += batch.num_graphs

            train_dist /= train_samples
            train_anatomy /= train_samples

            self.model.eval()
            val_dist = 0
            val_anatomy = 0
            val_samples = 0

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

                    pred_coords = self.model(features, edge_index, b)
                    pred_coords = self.procrustesAlign(pred_coords, coords_proj)
                    decoder_loss, dist = self.criterion(pred_coords, target)

                    val_dist += dist.item() * batch.num_graphs
                    val_anatomy += decoder_loss.item() * batch.num_graphs
                    val_samples += batch.num_graphs

            val_dist /= val_samples
            val_anatomy /= val_samples

            if self.scheduler is not None:
                self.scheduler.step(val_anatomy)

            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch: {epoch + 1} | "
                f"T-AL: {train_anatomy: .6f} | "
                f"T-D: {train_dist: .6f} | "
                f"V-AL: {val_anatomy: .6f} | "
                f"V-D: {val_dist: .6f} | "
                f"LR: {lr: .6f}"
            )

            if val_anatomy < self.best_loss - self.min_delta:
                self.best_loss = val_anatomy
                torch.save(self.model.state_dict(), self.model_save_path)

            if trial is not None:
                trial.report(val_anatomy, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            """
            self.early_stopper(val_loss, self.model)
            if self.early_stopper.stopping:
                print(f"Early Stopping at epoch {epoch + 1} / {self.num_epochs}")
                break
            """
        return train_anatomy, val_anatomy
