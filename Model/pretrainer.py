from itertools import cycle
from pathlib import Path

import optuna
import torch
from earlystopper import EarlyStopping
from torch import nn, optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        configs: dict,
        decoder_criterion: nn,
        generator_criterion: nn,
        decoder_model: nn.Module,
        generator_model: nn.Module,
        decoder_train_loader: DataLoader,
        decoder_val_loader: DataLoader,
        generator_train_loader: DataLoader,
        generator_val_loader: DataLoader,
        decoder_optimizer: optim,
        generator_optimizer: optim,
        decoder_scheduler,
        generator_scheduler,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.decoder_model = decoder_model.to(self.device)
        self.generator_model = generator_model.to(self.device)

        self.decoder_train_loader = decoder_train_loader
        self.decoder_val_loader = decoder_val_loader
        self.generator_train_loader = generator_train_loader
        self.generator_val_loader = generator_val_loader

        self.decoder_optimizer = decoder_optimizer
        self.decoder_scheduler = decoder_scheduler
        self.generator_optimizer = generator_optimizer
        self.generator_scheduler = generator_scheduler

        self.decoder_criterion = decoder_criterion
        self.generator_criterion = generator_criterion
        self.num_epochs = configs["num_epochs"]

        self.gen_model_save_path = (
            Path(configs["model_dir"]) / configs["generator_model_name"]
        )
        self.dec_model_save_path = (
            Path(configs["model_dir"]) / configs["decoder_model_name"]
        )

        self.gen_es = EarlyStopping(
            configs["es_patience"], configs["es_thresh"], self.gen_model_save_path
        )

        self.chol_row = torch.load(
            Path(configs["post_data_dir"]) / "cholrow.pt",
            weights_only=False,
        ).to(self.device, dtype=torch.float32)

        self.chol_col = torch.load(
            Path(configs["post_data_dir"]) / "cholcol.pt",
            weights_only=False,
        ).to(self.device, dtype=torch.float32)

        # Keeps matrices non-singular for stable training
        jitter = 1e-6
        self.cov_row = self.chol_row @ self.chol_row.T
        self.cov_row = self.cov_row + jitter * torch.eye(
            self.cov_row.shape[-1], device=self.device, dtype=self.cov_row.dtype
        )
        cov_col = self.chol_col @ self.chol_col.T
        cov_col = cov_col + jitter * torch.eye(
            cov_col.shape[-1], device=self.device, dtype=cov_col.dtype
        )

        self.col_inv = torch.linalg.inv(cov_col)
        self.logdet_col = torch.linalg.slogdet(cov_col).logabsdet

        next_idx = [0]
        for finger in range(5):
            pip = 2 + 4 * finger
            dip = 3 + 4 * finger
            tip = 4 + 4 * finger
            next_idx += [pip, dip, tip, tip]
        self.next_joint_idx = torch.tensor(
            next_idx, dtype=torch.long, device=self.device
        )

    def getFeatures(self, coords: torch.Tensor, eps: float = 1e-8):
        coords = coords.float()
        B = coords.shape[0]  # noqa: F841

        wrist = coords[:, 0:1, :]  # (B, 1, 3)
        scale = torch.linalg.norm(coords[:, 9] - coords[:, 0], dim=-1).clamp_min(
            eps
        )  # (B,)

        coords_norm = (coords - wrist) / scale[:, None, None]  # (B, 21, 3)

        joint_norms = torch.linalg.norm(coords, dim=-1).clamp_min(eps)  # (B, 21)
        dir_vectors = coords / joint_norms[..., None]  # (B, 21, 3)

        next_coords = coords[:, self.next_joint_idx, :]  # (B, 21, 3)
        dist_to_next = (
            torch.linalg.norm(next_coords - coords, dim=-1) / scale[:, None]
        )  # (B, 21)

        features = torch.cat(
            [coords_norm, dir_vectors, dist_to_next.unsqueeze(-1)], dim=-1
        )  # (B, 21, 7)

        return features

    def features(self, coords, left_kps, right_kps, noise=None, eps=1e-8):
        coords = coords.float()
        scale = torch.linalg.norm(coords[:, 9] - coords[:, 0], dim=-1).clamp_min(eps)
        coords_proj = (
            (coords - scale[:, None, None] * noise) if noise is not None else coords
        )
        features = self.getFeatures(coords_proj, eps)

        return features, coords_proj, scale

    def train(self, trial=None):
        for epoch in range(self.num_epochs):
            generator_train_iter = cycle(self.generator_train_loader)

            self.generator_model.train()
            self.decoder_model.train()
            decoder_train_loss = 0
            generator_train_loss = 0
            dec_samples = 0
            gen_samples = 0

            for decoder_batch, left_kps, right_kps in self.decoder_train_loader:
                gen_batch, gen_targets, gen_raw = next(generator_train_iter)

                decoder_batch = decoder_batch.to(self.device)
                left_kps = left_kps.to(self.device)
                right_kps = right_kps.to(self.device)

                gen_batch = gen_batch.to(self.device)
                gen_targets = gen_targets.to(self.device)
                gen_raw = gen_raw.to(self.device)

                # Start generator training
                generator_features = gen_batch.x.to(self.device, dtype=torch.float32)
                generator_edge_index = gen_batch.edge_index.to(self.device)
                generator_b = gen_batch.batch.to(self.device)

                self.generator_optimizer.zero_grad()

                gen_scale = torch.linalg.norm(
                    gen_targets[:, 9] - gen_targets[:, 0], dim=1
                )
                true_errors = (gen_targets - gen_raw) / gen_scale[:, None, None]

                mean_mat, scale = self.generator_model(
                    generator_features, generator_edge_index, generator_b
                )
                mean_mat = mean_mat.reshape(mean_mat.size(0), 21, 3)
                generator_loss = self.generator_criterion(
                    true_errors,
                    mean_mat,
                    scale,
                    self.cov_row,
                    self.col_inv,
                    self.logdet_col,
                    self.device,
                )

                generator_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.generator_model.parameters(), max_norm=1.0
                )
                self.generator_optimizer.step()

                generator_train_loss += generator_loss.item() * gen_batch.num_graphs
                gen_samples += gen_batch.num_graphs

                # Start decoder training
                decoder_coords = decoder_batch.x.to(
                    self.device, dtype=torch.float32
                ).view(decoder_batch.num_graphs, 21, 3)
                decoder_edge_index = decoder_batch.edge_index.to(self.device)
                decoder_b = decoder_batch.batch.to(self.device)

                self.decoder_optimizer.zero_grad()
                normalized_features, _, _ = self.features(
                    decoder_coords, left_kps, right_kps, None
                )  # Original 3D normalized features

                # Generate noise matrix and scales
                with torch.inference_mode():
                    mean_mat, scale = self.generator_model(
                        normalized_features, decoder_edge_index, decoder_b
                    )
                    mean_mat = mean_mat.reshape(mean_mat.size(0), 21, 3)
                    Z = torch.randn_like(mean_mat)
                    noise = mean_mat + torch.sqrt(scale)[:, None, None] * (
                        self.chol_row @ Z @ self.chol_col.T
                    )

                normalized_features, coords_proj, scale = self.features(
                    decoder_coords, left_kps, right_kps, noise
                )  # Distorted 3D normalized features with noise

                errors = self.decoder_model(
                    normalized_features, decoder_edge_index, decoder_b
                )

                errors = errors.view(errors.size(0), 21, 3)
                pred_coords = coords_proj + (scale[:, None, None] * errors)
                decoder_loss = self.decoder_criterion(pred_coords, decoder_coords)

                decoder_loss.backward()
                self.decoder_optimizer.step()
                decoder_train_loss += decoder_loss.item() * decoder_batch.num_graphs
                dec_samples += decoder_batch.num_graphs

            decoder_train_loss /= dec_samples
            generator_train_loss /= gen_samples

            self.generator_model.eval()
            self.decoder_model.eval()

            decoder_val_loss = 0
            generator_val_loss = 0
            dec_samples = 0
            gen_samples = 0

            # Decoder Validation
            with torch.inference_mode():
                for decoder_batch, left_kps, right_kps in self.decoder_val_loader:
                    decoder_batch = decoder_batch.to(self.device)
                    left_kps = left_kps.to(self.device)
                    right_kps = right_kps.to(self.device)

                    decoder_coords = decoder_batch.x.to(
                        self.device, dtype=torch.float32
                    ).view(decoder_batch.num_graphs, 21, 3)
                    decoder_edge_index = decoder_batch.edge_index.to(self.device)
                    decoder_b = decoder_batch.batch.to(self.device)

                    normalized_features, _, _ = self.features(
                        decoder_coords, left_kps, right_kps, None
                    )
                    mean_mat, scale = self.generator_model(
                        normalized_features, decoder_edge_index, decoder_b
                    )
                    mean_mat = mean_mat.reshape(mean_mat.size(0), 21, 3)
                    Z = torch.randn_like(mean_mat)
                    noise = mean_mat + torch.sqrt(scale)[:, None, None] * (
                        self.chol_row @ Z @ self.chol_col.T
                    )

                    normalized_features, coords_proj, scale = self.features(
                        decoder_coords, left_kps, right_kps, noise
                    )
                    errors = self.decoder_model(
                        normalized_features, decoder_edge_index, decoder_b
                    )
                    errors = errors.view(errors.size(0), 21, 3)
                    pred_coords = coords_proj + (scale[:, None, None] * errors)

                    decoder_loss = self.decoder_criterion(pred_coords, decoder_coords)
                    decoder_val_loss += decoder_loss.item() * decoder_batch.num_graphs
                    dec_samples += decoder_batch.num_graphs
            decoder_val_loss /= dec_samples

            # Generator Validation
            with torch.inference_mode():
                for generator_batch, gen_targets, gen_raw in self.generator_val_loader:
                    generator_batch = generator_batch.to(self.device)
                    gen_targets = gen_targets.to(self.device)
                    gen_raw = gen_raw.to(self.device)

                    generator_features = generator_batch.x.to(
                        self.device, dtype=torch.float32
                    )
                    generator_edge_index = generator_batch.edge_index.to(self.device)
                    generator_b = generator_batch.batch.to(self.device)

                    gen_scale = torch.linalg.norm(
                        gen_targets[:, 9] - gen_targets[:, 0], dim=1
                    )
                    true_errors = (gen_targets - gen_raw) / gen_scale[:, None, None]

                    mean_mat, scale = self.generator_model(
                        generator_features, generator_edge_index, generator_b
                    )
                    mean_mat = mean_mat.reshape(mean_mat.size(0), 21, 3)
                    generator_loss = self.generator_criterion(
                        true_errors,
                        mean_mat,
                        scale,
                        self.cov_row,
                        self.col_inv,
                        self.logdet_col,
                        self.device,
                    )
                    generator_val_loss += (
                        generator_loss.item() * generator_batch.num_graphs
                    )
                    gen_samples += generator_batch.num_graphs
            generator_val_loss /= gen_samples

            if self.decoder_scheduler is not None:
                self.decoder_scheduler.step(decoder_val_loss)

            if self.generator_scheduler is not None:
                self.generator_scheduler.step(generator_val_loss)

            current_dec_lr = self.decoder_optimizer.param_groups[0]["lr"]
            current_gen_lr = self.generator_optimizer.param_groups[0]["lr"]
            print(
                f"Epoch: {epoch + 1} | DecTL: {decoder_train_loss} | GenTL: {generator_train_loss} | DecVL: {decoder_val_loss} | GenVL: {generator_val_loss} | DecLR: {current_dec_lr: .5f} | GenLR: {current_gen_lr: .5f}"
            )

            if trial is not None:
                trial.report(decoder_val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            """
            Only use for post optuna fine tuning
            self.gen_es(generator_val_loss, self.generator_model)
            if self.gen_es.stopping:
                print(f"Generator Early Stopping at epoch {epoch} / {self.num_epochs}")
            """

        torch.save(self.decoder_model.state_dict(), self.dec_model_save_path)
        # Use for optuna hyperparameter selection
        torch.save(self.generator_model.state_dict(), self.gen_model_save_path)

        return decoder_train_loss, decoder_val_loss
