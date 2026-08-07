from pathlib import Path

import numpy as np
import optuna
import torch
from torch import device, nn, optim
from torch.utils.data import DataLoader
from utils import procrustesAlign


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        joint_predictor: nn.Module,
        configs: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim,
        scheduler: optim,
        criterion: nn,
        device: device,
    ):
        self.device = device
        self.mano_model = model
        self.joint_predictor = joint_predictor

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.num_epochs = configs["num_epochs"]

        decoder_stats = torch.load(
            Path(configs["stereo_data_dir"]) / "Training" / "Decoder" / "stats.pt",
            weights_only=False,
        )
        self.decoder_mean = decoder_stats[0].to(self.device, dtype=torch.float32)
        self.decoder_std = decoder_stats[1].to(self.device, dtype=torch.float32)

        generator_stats = torch.load(
            Path(configs["stereo_data_dir"]) / "Training" / "Generator" / "stats.pt",
            weights_only=False,
        )
        self.generator_mean = generator_stats[0].to(self.device, dtype=torch.float32)
        self.generator_std = generator_stats[1].to(self.device, dtype=torch.float32)

        self.mano_model_save_path = (
            Path(configs["model_dir"]) / f"{configs['model_name']}"
        )
        self.best_loss = np.inf
        self.min_delta = configs["es_thresh"]

        next_idx = [0]
        for finger in range(5):
            pip = 2 + 4 * finger
            dip = 3 + 4 * finger
            tip = 4 + 4 * finger
            next_idx += [pip, dip, tip, tip]
        self.next_joint_idx = torch.tensor(
            next_idx, dtype=torch.long, device=self.device
        )

    def constructFeatures(self, coords, eps: float = 1e-8):
        coords = coords.float()
        B = coords.shape[0]  # noqa: F841

        wrist = coords[:, 0:1, :]
        scale = torch.linalg.norm(coords[:, 9] - coords[:, 0], dim=-1).clamp_min(
            eps
        )  # (B,)

        coords_norm = (coords - wrist) / scale[:, None, None]

        joint_norms = torch.linalg.norm(coords, dim=-1).clamp_min(eps)
        dir_vectors = coords / joint_norms[..., None]

        next_coords = coords[:, self.next_joint_idx, :]
        dist_to_next = torch.linalg.norm(next_coords - coords, dim=-1) / scale[:, None]

        features = torch.cat(
            [coords_norm, dir_vectors, dist_to_next.unsqueeze(-1)], dim=-1
        )
        return features

    def getNewFeatures(self, old_features, coords):
        new_3D_feats = self.constructFeatures(coords)
        updated_features = old_features.clone()
        updated_features[:, :, :7] = new_3D_feats
        return updated_features

    def standardize(self, features, decoder: bool = True):
        mean = self.decoder_mean if decoder else self.generator_mean
        std = self.decoder_std if decoder else self.generator_std
        stand_feats = (features - mean) / std
        return stand_feats

    def train(self, trial):
        for epoch in range(self.num_epochs):
            self.joint_predictor.eval()
            self.mano_model.train()

            train_dist = 0
            train_loss = 0
            train_samples = 0

            for batch, target, coords_proj in self.train_loader:
                batch = batch.to(self.device)
                target = target.to(self.device)
                coords_proj = coords_proj.to(self.device)

                features = batch.x.to(self.device, dtype=torch.float32)
                orig_features = features.reshape(batch.num_graphs, 21, 19)
                features = self.standardize(orig_features)
                edge_index = batch.edge_index.to(self.device)
                b = batch.batch.to(self.device)

                self.optimizer.zero_grad()
                with torch.no_grad():
                    joint_pred_coords = self.joint_predictor(features, edge_index, b)

                updated_features = self.getNewFeatures(orig_features, joint_pred_coords)
                updated_features = self.standardize(updated_features, decoder=False)
                pred_coords = self.mano_model(updated_features, edge_index, b)
                pred_coords = procrustesAlign(pred_coords, joint_pred_coords)

                loss, dist = self.criterion(pred_coords, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.mano_model.parameters(), max_norm=1.0
                )
                self.optimizer.step()

                train_dist += dist.item() * batch.num_graphs
                train_loss += loss.item() * batch.num_graphs
                train_samples += batch.num_graphs

            train_dist /= train_samples
            train_loss /= train_samples

            self.mano_model.eval()
            val_dist = 0
            val_loss = 0
            val_samples = 0

            with torch.inference_mode():
                for batch, target, coords_proj in self.val_loader:
                    batch = batch.to(self.device)
                    target = target.to(self.device)
                    coords_proj = coords_proj.to(self.device)

                    features = batch.x.to(self.device, dtype=torch.float32)
                    orig_features = features.reshape(batch.num_graphs, 21, 19)
                    features = self.standardize(orig_features)
                    edge_index = batch.edge_index.to(self.device)
                    b = batch.batch.to(self.device)

                    joint_pred_coords = self.joint_predictor(features, edge_index, b)

                    updated_features = self.getNewFeatures(
                        orig_features, joint_pred_coords
                    )
                    updated_features = self.standardize(updated_features, decoder=False)
                    pred_coords = self.mano_model(updated_features, edge_index, b)
                    pred_coords = procrustesAlign(pred_coords, joint_pred_coords)

                    loss, dist = self.criterion(pred_coords, target)

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
                torch.save(self.mano_model.state_dict(), self.mano_model_save_path)

            if trial is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return (
            train_loss,
            val_loss,
            train_dist,
            val_dist,
        )
