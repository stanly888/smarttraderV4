
import torch
import torch.nn as nn

class PPOPolicy(nn.Module):
    def __init__(self, input_dim=10, output_dim=3):  # 假設狀態維度包含技術指標
        super(PPOPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )
        self.leverage_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 1, scaled later to leverage
        )

    def forward(self, x):
        action_probs = self.net(x)
        leverage = self.leverage_net(x) * 20  # Scale to 1~20x leverage
        return action_probs, leverage


class TP_SL_Model(nn.Module):
    def __init__(self, input_dim=10):
        super(TP_SL_Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)
