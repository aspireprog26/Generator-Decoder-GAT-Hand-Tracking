import sys
import torch
import optuna
import joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import earlystopper as es
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/mrtcloud-1/Documents/Hand-Tracking-2/Dataset")
from optimizedataset import DatasetOptimizer
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
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.num_epochs = configs["num_epochs"]
        min_delta = configs["es_thresh"]
        patience = configs["es_patience"]

        model_save_path = Path(configs["model_dir"]) / configs["model_name"]
        self.cluster_dir = configs["post_data_dir"] / "Clusters"

        self.early_stopper = es.EarlyStopping(patience, min_delta, model_save_path)
        self.features = DatasetOptimizer(mp = False).getFeatures
        self.clusterer = joblib.load(
            configs["post_data_dir"] / "Clusters" / "poseclusterer.joblib"
        )

        self.chol_row = np.load(self.cluster_dir / "cholrow.npy")
        self.chol_col = np.load(self.cluster_dir / "cholcol.npy")

    def generateErrors(self, labels):
        errors = []
        for label in labels.tolist():
            mean_mat = self.cluster_dir / f"meanmat{label}.npy"
            Z = np.random.randn(*mean_mat.shape)
            sample = mean_mat + self.chol_row @ Z @ self.chol_col.T
            errors.append(sample)

        return torch.stack(errors, dim=0)

    def train(self, trial = None):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            for batch in self.train_loader:
                batch = batch.to(self.device)
                coords = batch.x
                edge_index = batch.edge_index
                b = batch.batch

                self.optimizer.zero_grad()
                
                normalized_coords, coords_proj, scale = self.features(coords.detach().cpu().numpy(), None, True)                # Original 3D normalizaed features
                
                noise = self.generateErrors()
                normalized_coords, coords_proj, scale = self.features(coords.detach().cpu().numpy(), noise, True)               # Distorted 3D normalized features with nouse
                normalized_coords = torch.tensor(normalized_coords).to(self.device)
                coords_proj = torch.tensor(coords_proj).to(self.device)
                scale = torch.tensor(scale).to(self.device)

                errors = self.model(normalized_coords, edge_index, b)
                pred = coords_proj + (scale * errors)
                loss = self.criterion(pred, target)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)

            self.model.eval()
            val_loss = 0
            with torch.inference_mode():
                for batch, target in self.val_loader:
                    batch = batch.to(self.device)
                    target = target.to(self.device)

                    features = batch.x
                    edge_index = batch.edge_index
                    b = batch.batch

                    output = self.model(features, edge_index, b)
                    loss = self.criterion(output, target)
                    val_loss += loss.item()
            val_loss /= len(self.val_loader)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch: {epoch + 1} | Train Loss: {train_loss} | Val Loss: {val_loss} | LR: {current_lr}"
            )

            if trial is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            self.early_stopper(val_loss, self.model)
            if self.early_stopper.stopping:
                print(f"Early Stopping at epoch {epoch + 1} / {self.num_epochs}")
                break

        return val_loss
