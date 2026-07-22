import sys
import torch
import optuna
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import earlystopper as es
from itertools import cycle
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/mrtcloud-1/Documents/Hand-Tracking-2/Dataset")
from optimizedataset import DatasetOptimizer


class Trainer:
    def __init__(
        self,
        configs: dict,
        criterion: nn,
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

        self.criterion = criterion
        self.num_epochs = configs["num_epochs"]
        min_delta = configs["es_thresh"]
        patience = configs["es_patience"]

        gen_model_save_path = (
            Path(configs["model_dir"]) / configs["generator_model_name"]
        )
        dec_model_save_path = Path(configs["model_dir"]) / configs["decoder_model_name"]
        self.early_stopper = es.EarlyStopping(
            patience, min_delta, dec_model_save_path, gen_model_save_path
        )

        self.features = DatasetOptimizer(mp=False).getFeatures
        self.chol_row = torch.load(Path(configs["post_data_dir"]) / "cholrow.pt")
        self.chol_col = torch.load(Path(configs["post_data_dir"]) / "cholcol.pt")

    def train(self, trial=None):
        for epoch in range(self.num_epochs):
            generator_train_iter = cycle(self.generator_train_loader)

            self.generator_model.train()
            self.decoder_model.train()
            decoder_train_loss = 0
            generator_train_loss = 0
            gen_samples = 0

            for decoder_batch in self.decoder_train_loader:
                gen_batch, gen_targets, gen_raw = next(generator_train_iter)

                decoder_batch = decoder_batch.to(self.device)
                gen_batch = gen_batch.to(self.device)
                gen_targets = gen_targets.to(self.device)
                gen_raw = gen_raw.to(self.device)

                # Start generator training
                generator_features = gen_batch.x
                generator_edge_index = gen_batch.edge_index
                generator_b = gen_batch.batch

                self.generator_optimizer.zero_grad()

                gen_scale = torch.linalg.norm(gen_targets[:, 9] - gen_targets[:, 0])
                true_errors = (gen_targets - gen_raw) / gen_scale

                pred_errors = self.generator_model(
                    generator_features, generator_edge_index, generator_b
                )
                generator_loss = self.criterion(pred_errors, true_errors)

                generator_loss.backward()
                self.generator_optimizer.step()

                batch_size = gen_targets.size(0)
                generator_train_loss += generator_loss.item() * batch_size
                gen_samples += batch_size

                # Start decoder training
                decoder_coords = decoder_batch.x
                decoder_edge_index = decoder_batch.edge_index
                decoder_b = decoder_batch.batch

                self.decoder_optimizer.zero_grad()

                normalized_coords, _, _ = self.features(
                    decoder_coords.detach().cpu().numpy(), None, True
                )  # Original 3D normalized features
                normalized_coords = torch.tensor(normalized_coords).to(self.device)

                # Generate noise matrix and scales
                with torch.inference_mode():
                    mean_mat, scale_col, scale_row = self.generator_model(
                        normalized_coords, decoder_edge_index, decoder_b
                    )
                    mean_mat = mean_mat.reshape(21, 3)
                    Z = torch.rand_like(mean_mat)
                    noise = mean_mat + (scale_col * scale_row) * (
                        self.chol_row @ Z @ self.chol_col
                    )

                normalized_coords, coords_proj, scale = self.features(
                    decoder_coords.detach().cpu().numpy(), noise, True
                )  # Distorted 3D normalized features with noise

                normalized_coords = torch.tensor(normalized_coords).to(self.device)
                coords_proj = torch.tensor(coords_proj).to(self.device)
                scale = torch.tensor(scale).to(self.device)

                errors = self.decoder_model(
                    normalized_coords, decoder_edge_index, decoder_b
                )
                pred_coords = coords_proj + (scale * errors)
                decoder_loss = self.criterion(pred_coords, decoder_coords)

                decoder_loss.backward()
                self.decoder_optimizer.step()
                decoder_train_loss += decoder_loss.item()

            decoder_train_loss /= len(self.decoder_train_loader)
            generator_train_loss /= gen_samples

            self.generator_model.eval()
            self.decoder_model.eval()
            decoder_val_loss = 0
            generator_val_loss = 0
            gen_samples = 0

            # Decoder Validation
            with torch.inference_mode():
                for decoder_batch in self.decoder_val_loader:
                    decoder_batch = decoder_batch.to(self.device)
                    decoder_coords = decoder_batch.x
                    decoder_edge_index = decoder_batch.edge_index
                    decoder_b = decoder_batch.batch

                    normalized_coords, _, _ = self.features(
                        decoder_coords.detach().cpu().numpy(), None, True
                    )  # Original 3D normalized features
                    normalized_coords = torch.tensor(normalized_coords).to(self.device)

                    with torch.inference_mode():
                        mean_mat, scale_col, scale_row = self.generator_model(
                            normalized_coords, decoder_edge_index, decoder_b
                        )
                        mean_mat = mean_mat.reshape(21, 3)
                        Z = torch.rand_like(mean_mat)
                        noise = mean_mat + (scale_col * scale_row) * (
                            self.chol_row @ Z @ self.chol_col
                        )

                    normalized_coords, coords_proj, scale = self.features(
                        decoder_coords.detach().cpu().numpy(), noise, True
                    )  # Distorted 3D normalized features with noise

                    normalized_coords = torch.tensor(normalized_coords).to(self.device)
                    coords_proj = torch.tensor(coords_proj).to(self.device)
                    scale = torch.tensor(scale).to(self.device)

                    errors = self.decoder_model(
                        normalized_coords, decoder_edge_index, decoder_b
                    )
                    pred_coords = coords_proj + (scale * errors)
                    decoder_loss = self.criterion(pred_coords, decoder_coords)
                    decoder_val_loss += decoder_loss.item()
            decoder_val_loss /= len(self.decoder_val_loader)

            # Generator Validation
            with torch.inference_mode():
                for generator_batch, gen_targets, gen_raw in self.generator_val_loader:
                    generator_features = generator_batch.x
                    generator_edge_index = generator_batch.edge_index
                    generator_b = generator_batch.batch

                    gen_scale = torch.linalg.norm(gen_targets[:, 9] - gen_targets[:, 0])
                    true_errors = (gen_targets - gen_raw) / gen_scale

                    pred_errors = self.generator_model(
                        generator_features, generator_edge_index, generator_b
                    )
                    generator_loss = self.criterion(pred_errors, true_errors)
                    batch_size = gen_targets.size(0)
                    generator_val_loss += generator_loss.item() * batch_size
                    gen_samples += batch_size
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

            self.early_stopper(
                decoder_val_loss, self.decoder_model, self.generator_model
            )
            if self.early_stopper.stopping:
                print(f"Early Stopping at epoch {epoch + 1} / {self.num_epochs}")
                break

        return decoder_train_loss, decoder_val_loss
