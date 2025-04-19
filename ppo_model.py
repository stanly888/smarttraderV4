# ppo_model.py
import torch
import torch.nn as nn
import os

class UnifiedRLModel(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64):  # ✅ input_dim 升級為 20
        super(UnifiedRLModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_dim, 2)   # Long / Short
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy_head(x), self.value_head(x)

    def act(self, x):
        logits, value = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def evaluate(self, x, action):
        logits, value = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value

def save_model(model, path="ppo_model.pt"):
    torch.save(model.state_dict(), path)
    print(f"✅ 模型已儲存：{path}")

def load_model_if_exists(model, path="ppo_model.pt"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"✅ 模型已載入：{path}")
    else:
        print("⚠️ 未找到模型檔案，將使用未訓練的初始化模型")
