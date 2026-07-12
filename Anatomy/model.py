import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GraphAttentionNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gat1 = GATConv(in_channels = input_size, out_channels = 8, heads = 4, dropout = 0.2)       # 4 attention heads * 16 out channels -> 32 out
        self.gat2 = GATConv(in_channels = 32, out_channels = 32, heads = 1, dropout = 0.2)              # 1 attention head * 32 out channels -> 32 out
        self.layer_norm = nn.LayerNorm(32)
        self.elu = nn.ELU()

    def forward(self, features, edge_index, batch):
        features = self.gat1(features, edge_index)          
        features = self.gat2(features, edge_index)
        batch_size = batch.max().item() + 1
        features = features.view(batch_size, -1)                                                         # (B, 21, 32) -> (B, 672)
        return features
    
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
        self.dropout1 = nn.Dropout(p = 0.2)

        self.fc2 = nn.Linear(128, 64)
        self.layer_norm2 = nn.LayerNorm(64)
        self.dropout2 = nn.Dropout(p = 0.2)

        self.out = nn.Linear(64, output_size)
        
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.layer_norm1(self.gelu(self.fc1(x)))
        x = self.dropout1(x)
        x = self.layer_norm2(self.gelu(self.fc2(x)))
        x = self.dropout2(x)
        out = self.out(x)
        out = self.softmax(out, dim = -1)
        return out
    
class AnatomyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.gat = GraphAttentionNet(input_size)                                        # (B, 21, 3) -> (B, 672)
        self.encoder = Encoder(672, 672)                                                # (B, 672) -> (B, 672)
        self.gated_fusion = GatedFusion(672, 672)                                       # (B, 672) -> (B, 672)
        self.fusion_fc = nn.Linear(672, 256)                                            # (B, 672) -> (B, 256)
        self.regressor = Regressor(256, output_size)                                    # (B, 256) -> (B, 2)

    def forward(self, features_left, features_right, edge_index_left, edge_index_right, batch_left, batch_right):
        left_gat = self.gat(features_left, edge_index_left, batch_left)                 # left hand joint graph embedding pooled vector
        right_gat = self.gat(features_right, edge_index_right, batch_right)             # right hand joint graph embedding pooled vector
        
        left_encoder = self.encoder(left_gat)                                           # latent left embedding
        right_encoder = self.encoder(right_gat)                                         # latent right embedding

        fused_encoders = self.gated_fusion(left_encoder, right_encoder)                 # fuse the left and right latent embeddings
        fused_fc = self.fusion_fc(fused_encoders)                                       # linearly project fused vector 

        out = self.regressor(fused_fc)                                                  # regress fused vector
        return out