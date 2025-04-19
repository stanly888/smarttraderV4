# ppo_model.py
import torch
import torch.nn as nn
import os

class UnifiedRLModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super(UnifiedRLModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, 2)    # logits for Long / Short
        self.value_head = nn.Linear(hidden_dim, 1)     # Critic value
        self.tp_head = nn.Linear(hidden_dim, 1)        # Take Profit %
        self.sl_head = nn.Linear(hidden_dim, 1)        # Stop Loss %
        self.lev_head = nn.Linear(hidden_dim, 1)       # Leverage 倍數

    def forward(self, x):
        shared_out = self.shared(x)
        logits = self.policy_head(shared_out)
        value = self.value_head(shared_out)
        tp_out = self.tp_head(shared_out)
        sl_out = self.sl_head(shared_out)
        lev_out = self.lev_head(shared_out)
        return logits, value, tp_out, sl_out, lev_out

    def act(self, x):
        logits, value, tp_out, sl_out, lev_out = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        tp = torch.sigmoid(tp_out) * 3.5       # TP %：0 ~ 3.5
        sl = torch.sigmoid(sl_out) * 2.0       # SL %：0 ~ 2.0
        lev = torch.sigmoid(lev_out) * 9 + 1   # Leverage：1 ~ 10 倍

        return action.item(), dist.log_prob(action), value, tp.item(), sl.item(), lev.item()

    def evaluate(self, x, action):
        logits, value, _, _, _ = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value

# ✅ 儲存模型
def save_model(model, path="ppo_model.pt"):
    torch.save(model.state_dict(), path)
    print(f"✅ 模型已儲存：{path}")

# ✅ 載入模型（若存在）
def load_model_if_exists(model, path="ppo_model.pt"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"✅ 模型已載入：{path}")
    else:
        print("⚠️ 未找到模型檔案，將使用未訓練的初始化模型")
