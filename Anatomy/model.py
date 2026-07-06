import torch
import torch.nn as nn
import torch.nn.functional as F
class AnatomyRegression(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(AnatomyRegression, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(p = 0.4)

        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.dropout2 = nn.Dropout(p = 0.4)

        self.out = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
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
                self.tanh(finger[:, 12:18]),              # activations for dot product and sign
                finger[:, 18:20]                          # ratios left unchanged
            ]
            new_fingers.append(torch.cat(parts, dim = -1))

        out = torch.cat(new_fingers, dim = -1)
        return out