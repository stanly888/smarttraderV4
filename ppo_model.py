# ppo_model.py
import torch
import torch.nn as nn

class PPOModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super(PPOModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, 2)  # 2 類別：Long / Short
        self.value_head = nn.Linear(hidden_dim, 1)   # 狀態價值

    def forward(self, x):
        x = self.shared(x)
        return self.policy_head(x), self.value_head(x)
