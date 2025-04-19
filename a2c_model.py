# a2c_model.py
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, action_dim: int = 2):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        # 輸出層（與 PPO 統一）
        self.actor = nn.Linear(hidden_dim, action_dim)  # logits
        self.critic = nn.Linear(hidden_dim, 1)          # value
        self.tp_head = nn.Linear(hidden_dim, 1)         # TP %
        self.sl_head = nn.Linear(hidden_dim, 1)         # SL %
        self.lev_head = nn.Linear(hidden_dim, 1)        # Leverage

    def forward(self, x):
        shared_out = self.shared(x)
        logits = self.actor(shared_out)
        value = self.critic(shared_out)
        tp_out = self.tp_head(shared_out)
        sl_out = self.sl_head(shared_out)
        lev_out = self.lev_head(shared_out)
        return logits, value, tp_out, sl_out, lev_out