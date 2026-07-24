import math

import torch
from torch import nn
from torch_geometric.nn import GATConv


class GraphAttentionNet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        out_channels_1 = hidden_size // 8

        self.gat1 = GATConv(
            in_channels=input_size,
            out_channels=out_channels_1,
            heads=8,
            dropout=dropout,
        )
        self.gat2 = GATConv(
            in_channels=hidden_size, out_channels=hidden_size, heads=1, dropout=dropout
        )
        self.elu = nn.ELU()

    def forward(self, features, edge_index, batch):
        if features.dim() == 3:
            features = features.flatten(0, 1)  # Flatten to shape (B * 21, 7)

        features = self.elu(self.gat1(features, edge_index))
        features = self.elu(self.gat2(features, edge_index))
        batch_size = batch.max().item() + 1
        features = features.view(batch_size, -1)  # (B, 21, hidden) -> (B, 21 * hidden)
        return features


class Regressor(nn.Module):
    def __init__(self, input_size, output_size, dropout, generator):
        super().__init__()
        hidden1 = int((2 ** math.floor(math.log2(input_size))) / 2)
        hidden2 = hidden1 // 2 if generator else hidden1 // 4

        self.fc1 = nn.Linear(input_size, hidden1)
        self.layer_norm1 = nn.LayerNorm(hidden1)
        self.dropout1 = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.layer_norm2 = nn.LayerNorm(hidden2)
        self.dropout2 = nn.Dropout(p=dropout)

        self.out = nn.Linear(hidden2, output_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.layer_norm1(self.gelu(self.fc1(x)))
        x = self.dropout1(x)

        x = self.layer_norm2(self.gelu(self.fc2(x)))
        x = self.dropout2(x)

        out = self.out(x)
        return out


class AnatomyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, generator: bool):
        super().__init__()

        self.gat = GraphAttentionNet(input_size, hidden_size, dropout)
        self.regressor = Regressor(21 * hidden_size, output_size, dropout, generator)
        self.softplus = nn.Softplus()
        self.generator = generator

        if self.generator:
            with torch.no_grad():
                # Encourages scale to start near 1 since softplus^-1(-1) = 0.5413
                self.regressor.out.bias[63] = 0.5413

    def forward(self, features, edge_index, batch):
        gat = self.gat(features, edge_index, batch)
        out = self.regressor(gat)

        if self.generator:
            mean_mat = out[:, :63]
            # Add term to keep scale >> 0 to avoid exploding variance NLL computations
            scale = self.softplus(out[:, 63]) + 1e-1

            return mean_mat, scale
        else:
            return out
