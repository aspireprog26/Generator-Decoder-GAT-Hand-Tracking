import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GraphAttentionNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv1 = GATConv(in_channels = input_size, out_channels = 16, heads = 4, dropout = 0.1)      # 4 attention heads * 16 out channels -> 64 out
        self.conv2 = GATConv(in_channels = 64, out_channels = 64, heads = 1, dropout = 0.1)              # 1 attention head * 64 out channels -> 64 out
        self.layer_norm = nn.LayerNorm(64)
        self.elu = nn.ELU()

    def forward(self, coords, edge_index, batch):
        coords = self.elu(self.layer_norm(self.conv1(coords, edge_index)))           
        coords = self.conv2(coords, edge_index)
        pooled_coords = global_mean_pool(coords, batch)                                                  # (B, N, 64) -> (B, 64)
        return pooled_coords
    
class Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)
        self.fc2 = nn.Linear(output_size, output_size)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.layer_norm(self.gelu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
class GatedFusion(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.hidden1_fc = nn.Linear(input_size, output_size)
        self.hidden2_fc = nn.Linear(input_size, output_size)
        self.concat_fc = nn.Linear(input_size * 2, output_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh(self.hidden1_fc(x1))
        h2 = self.tanh(self.hidden2_fc(x2))
        x_concat = torch.cat([x1, x2], dim = -1)
        z = self.sigmoid(self.concat_fc(x_concat))
        h = (z * h1) + (1 - z) * h2
        return h
    
class Regressor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.layer_norm1 = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(p = 0.1)

        self.fc2 = nn.Linear(128, 256)
        self.layer_norm2 = nn.LayerNorm(256)
        self.dropout2 = nn.Dropout(p = 0.1)

        self.out = nn.Linear(256, output_size)
        
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.layer_norm1(self.gelu(self.fc1(x)))
        x = self.dropout1(x)
        x = self.layer_norm2(self.gelu(self.fc2(x)))
        x = self.dropout2(x)
        out = self.out(x)
        
        fingers = torch.chunk(out, 5, dim = -1)
        new_fingers = []

        for finger in fingers:
            parts = [
                F.normalize(finger[:, 0:3], dim = -1),    # CMC vector
                F.normalize(finger[:, 3:6], dim = -1),    # MCP vector
                F.normalize(finger[:, 6:9], dim = -1),    # IP vector
                F.normalize(finger[:, 9:12], dim = -1),   # TIP vector
                self.tanh(finger[:, 12:15]),               # activations for dot product 
                self.tanh(finger[:, 15:18]),              # activations for sign
                finger[:, 18:20]                          # ratios left unchanged
            ]
            new_fingers.append(torch.cat(parts, dim = -1))

        out = torch.cat(new_fingers, dim = -1)
        return out
    
class AnatomyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.gat = GraphAttentionNet(input_size)                                        # (B, 21, 2) -> (B, 64)
        self.encoder = Encoder(64, 64)                                                  # (B, 64) -> (B, 64)
        self.gated_fusion = GatedFusion(64, 64)                                         # (B, 64) -> (B, 64)
        self.fusion_fc = nn.Linear(64, 64)                                              # (B, 64) -> (B, 64)
        self.regressor = Regressor(64, output_size)                                     # (B, 64) -> (B, 100)

    def forward(self, coords_left, coords_right, edge_index_left, edge_index_right, batch_left, batch_right):
        left_gat = self.gat(coords_left, edge_index_left, batch_left)                   # left hand joint graph embedding pooled vector
        right_gat = self.gat(coords_right, edge_index_right, batch_right)               # right hand joint graph embedding pooled vector
        
        left_encoder = self.encoder(left_gat)                                           # latent left embedding
        right_encoder = self.encoder(right_gat)                                         # latent right embedding

        fused_encoders = self.gated_fusion(left_encoder, right_encoder)                 # fuse the left and right latent embeddings
        fused_fc = self.fusion_fc(fused_encoders)                                       # linearly project fused vector 

        out = self.regressor(fused_fc)                                                  # regress fused vector
        return out