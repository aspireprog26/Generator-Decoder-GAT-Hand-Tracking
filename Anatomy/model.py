import torch.nn as nn

class AnatomyRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AnatomyRegression, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(p = 0.2)

        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.dropout2 = nn.Dropout(p = 0.2)

        self.out = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        out = self.out(x)

        