import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, action_dim: int = 3):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared_out = self.shared(x)
        logits = self.actor(shared_out)
        value = self.critic(shared_out)
        return logits, value
