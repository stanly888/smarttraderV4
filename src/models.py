import torch.nn as nn
import torch

class PPOPolicy(nn.Module):
    def __init__(self, input_dim=10, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, output_dim), nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

class TP_SL_Model(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.model(x)
