# a2c_model.py
import torch
import torch.nn as nn

class A2CNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, action_dim=2):
        super(A2CNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.shared(x)
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value
