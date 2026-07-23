import torch
import optuna
from typing import Optional
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from itertools import cycle
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

        self.chol_row = torch.load(
            Path(configs["post_data_dir"]) / "cholrow.pt",
            weights_only=False,
        ).to(self.device)

        self.chol_col = torch.load(
            Path(configs["post_data_dir"]) / "cholcol.pt",
            weights_only=False,
        ).to(self.device)

        self.cov_row = self.chol_row @ self.chol_row.T
        cov_col = self.chol_col @ self.chol_col.T

        self.col_inv = torch.linalg.inv(cov_col)
        self.logdet_col = torch.linalg.slogdet(cov_col).logabsdet

    def features(
        self,
        c: torch.Tensor,
        n: Optional[torch.Tensor],
        batch: bool = False,
        eps: float = 1e-8,
    ):

        mcp_idx = torch.tensor([1, 5, 9, 13, 17], device=c.device)
        pip_idx = torch.tensor([2, 6, 10, 14, 18], device=c.device)
        dip_idx = torch.tensor([3, 7, 11, 15, 19], device=c.device)
        tip_idx = torch.tensor([4, 8, 12, 16, 20], device=c.device)

        def _one(coords: torch.Tensor, noise: Optional[torch.Tensor]):
            coords = coords.to(dtype=torch.float32)

            if noise is not None:
                noise = noise.to(device=coords.device, dtype=coords.dtype)

            # Original scale used before projection
            scale0 = torch.linalg.norm(coords[9] - coords[0])

            coords_proj = coords - scale0 * noise if noise is not None else coords
            scale = torch.linalg.norm(coords_proj[9] - coords_proj[0]).clamp_min(eps)

            # 3 normalized xyz features
            coords_proj_norm = (coords_proj - coords_proj[0]) / scale

            # 3 unit-vector features + 1 length feature per node
            unit_vecs = coords_proj / torch.linalg.norm(
                coords_proj, dim=-1, keepdim=True
            ).clamp_min(eps)

            lengths = torch.zeros((21, 1), device=coords.device, dtype=coords.dtype)

            lengths[mcp_idx, 0] = (
                torch.linalg.norm(coords_proj[pip_idx] - coords_proj[mcp_idx], dim=-1)
                / scale
            )
            lengths[pip_idx, 0] = (
                torch.linalg.norm(coords_proj[dip_idx] - coords_proj[pip_idx], dim=-1)
                / scale
            )
            lengths[dip_idx, 0] = (
                torch.linalg.norm(coords_proj[tip_idx] - coords_proj[dip_idx], dim=-1)
                / scale
            )
            # wrist and tips stay at 0

            features = torch.cat(
                [coords_proj_norm, unit_vecs, lengths], dim=-1
            )  # (21, 7)
            return features, coords_proj, scale

        if not batch:
            return _one(c, n)

        coords = c.to(dtype=torch.float32)
        noise = None if n is None else n.to(device=coords.device, dtype=coords.dtype)

        # Batched version
        scale0 = torch.linalg.norm(coords[:, 9] - coords[:, 0], dim=-1)  # (B,)
        coords_proj = (
            coords - scale0[:, None, None] * noise if noise is not None else coords
        )

        scale = torch.linalg.norm(
            coords_proj[:, 9] - coords_proj[:, 0], dim=-1
        ).clamp_min(eps)  # (B,)

        coords_proj_norm = (coords_proj - coords_proj[:, 0:1, :]) / scale[:, None, None]
        unit_vecs = coords_proj / torch.linalg.norm(
            coords_proj, dim=-1, keepdim=True
        ).clamp_min(eps)

        B = coords.shape[0]
        lengths = torch.zeros((B, 21, 1), device=coords.device, dtype=coords.dtype)

        lengths[:, mcp_idx, 0] = (
            torch.linalg.norm(coords_proj[:, pip_idx] - coords_proj[:, mcp_idx], dim=-1)
            / scale
        )
        lengths[:, pip_idx, 0] = (
            torch.linalg.norm(coords_proj[:, dip_idx] - coords_proj[:, pip_idx], dim=-1)
            / scale
        )
        lengths[:, dip_idx, 0] = (
            torch.linalg.norm(coords_proj[:, tip_idx] - coords_proj[:, dip_idx], dim=-1)
            / scale
        )

        features = torch.cat(
            [coords_proj_norm, unit_vecs, lengths], dim=-1
        )  # (B, 21, 7)
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

            for decoder_batch in self.decoder_train_loader:
                gen_batch, gen_targets, gen_raw = next(generator_train_iter)

                decoder_batch = decoder_batch.to(self.device)
                gen_batch = gen_batch.to(self.device)
                gen_targets = gen_targets.to(self.device)
                gen_raw = gen_raw.to(self.device)

                # Start generator training
                generator_features = gen_batch.x.float()
                generator_edge_index = gen_batch.edge_index
                generator_b = gen_batch.batch

                self.generator_optimizer.zero_grad()

                gen_scale = torch.linalg.norm(
                    gen_targets[:, 9] - gen_targets[:, 0], dim=1
                )
                true_errors = (gen_targets - gen_raw) / gen_scale[:, None, None]

                mean_mat, scale = self.generator_model(
                    generator_features, generator_edge_index, generator_b
                )
                mean_mat = mean_mat.reshape(gen_batch.num_graphs, 21, 3)
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
                self.generator_optimizer.step()

                generator_train_loss += generator_loss.item() * gen_batch.num_graphs
                gen_samples += gen_batch.num_graphs

                # Start decoder training
                decoder_coords = decoder_batch.x.float()
                decoder_edge_index = decoder_batch.edge_index
                decoder_b = decoder_batch.batch

                self.decoder_optimizer.zero_grad()
                normalized_coords, _, _ = self.features(
                    decoder_coords, None, True
                )  # Original 3D normalized features

                # Generate noise matrix and scales
                with torch.inference_mode():
                    mean_mat, scale = self.generator_model(
                        normalized_coords, decoder_edge_index, decoder_b
                    )
                    mean_mat = mean_mat.reshape(decoder_batch.num_graphs, 21, 3)
                    Z = torch.randn_like(mean_mat)
                    noise = mean_mat + torch.sqrt(scale)[:, None, None] * (
                        self.chol_row @ Z @ self.chol_col.T
                    )

                normalized_coords, coords_proj, scale = self.features(
                    decoder_coords, noise, True
                )  # Distorted 3D normalized features with noise

                errors = self.decoder_model(
                    normalized_coords, decoder_edge_index, decoder_b
                )

                pred_coords = coords_proj + (scale * errors)
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
                for decoder_batch in self.decoder_val_loader:
                    decoder_batch = decoder_batch.to(self.device)
                    decoder_coords = decoder_batch.x.float()
                    decoder_edge_index = decoder_batch.edge_index
                    decoder_b = decoder_batch.batch

                    normalized_coords, _, _ = self.features(decoder_coords, None, True)
                    mean_mat, scale = self.generator_model(
                        normalized_coords, decoder_edge_index, decoder_b
                    )
                    mean_mat = mean_mat.reshape(decoder_batch.num_graphs, 21, 3)
                    Z = torch.randn_like(mean_mat)
                    noise = mean_mat + torch.sqrt(scale)[:, None, None] * (
                        self.chol_row @ Z @ self.chol_col.T
                    )

                    normalized_coords, coords_proj, scale = self.features(
                        decoder_coords, noise, True
                    )
                    errors = self.decoder_model(
                        normalized_coords, decoder_edge_index, decoder_b
                    )
                    pred_coords = coords_proj + (scale * errors)
                    decoder_loss = self.decoder_criterion(pred_coords, decoder_coords)
                    decoder_val_loss += decoder_loss.item() * decoder_batch.num_graphs
                    dec_samples += decoder_batch.num_graphs
            decoder_val_loss /= dec_samples

            # Generator Validation
            with torch.inference_mode():
                for generator_batch, gen_targets, gen_raw in self.generator_val_loader:
                    generator_features = generator_batch.x.float()
                    generator_edge_index = generator_batch.edge_index
                    generator_b = generator_batch.batch

                    gen_scale = torch.linalg.norm(
                        gen_targets[:, 9] - gen_targets[:, 0], dim=1
                    )
                    true_errors = (gen_targets - gen_raw) / gen_scale[:, None, None]

                    mean_mat, scale = self.generator_model(
                        generator_features, generator_edge_index, generator_b
                    )
                    mean_mat = mean_mat.reshape(generator_batch.num_graphs, 21, 3)
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
                f"Epoch: {epoch + 1} | DecTL: {decoder_train_loss} | GenTL: {generator_train_loss} | DecVL: {decoder_val_loss} | GenVL: {generator_val_loss} | DecLR: {current_dec_lr} | GenLR: {current_gen_lr}"
            )

            if trial is not None:
                trial.report(decoder_val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        torch.save(self.decoder_model.state_dict(), self.dec_model_save_path)
        torch.save(self.generator_model.state_dict(), self.gen_model_save_path)

        return decoder_train_loss, decoder_val_loss
