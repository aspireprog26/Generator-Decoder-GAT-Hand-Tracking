import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GraphAttentionNet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        out_channels_1 = hidden_size / 8

        self.gat1 = GATConv(
            in_channels=input_size,
            out_channels=out_channels_1,
            heads=8,
            dropout=dropout,
        )
        self.gat2 = GATConv(
            in_channels=hidden_size, out_channels=hidden_size, heads=1, dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.elu = nn.ELU()
        self.out = nn.Linear(hidden_size * 21, (hidden_size * 21) / 2)

    def forward(self, features, edge_index, batch):
        features = self.gat1(features, edge_index)
        features = self.gat2(features, edge_index)
        batch_size = batch.max().item() + 1
        features = features.view(batch_size, -1)
        out = self.out(features)
        return out


class Regressor(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        hidden1 = torch.floor(torch.log2(input_size))

        self.fc1 = nn.Linear(input_size, hidden1)
        self.layer_norm1 = nn.LayerNorm(hidden1)
        self.dropout1 = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(hidden1 / 2, hidden1 / 4)
        self.layer_norm2 = nn.LayerNorm(64)
        self.dropout2 = nn.Dropout(p=dropout)

        self.out = nn.Linear(hidden1 / 4, output_size)
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
        self.regressor = Regressor(1344, output_size, dropout)
        self.softplus = nn.Softplus()
        self.generator = generator

    def forward(self, features, edge_index, batch):
        gat = self.gat(features, edge_index, batch)
        out = self.regressor(gat)

        if self.generator:
            errors = out[:, :63]
            row_scale = self.softplus(out[:, 63])
            col_scale = self.softplus(out[:, 64])

            return errors, row_scale, col_scale
        else:
            return out
