# a2c_model.py
import torch
import torch.nn as nn
import os

class ActorCritic(nn.Module):
    def __init__(self, input_dim: int = 33, hidden_dim: int = 64, action_dim: int = 2):  # ✅ input_dim=33
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        # 與 PPO 統一的輸出層
        self.actor = nn.Linear(hidden_dim, action_dim)  # logits（Long / Short）
        self.critic = nn.Linear(hidden_dim, 1)          # 狀態價值
        self.tp_head = nn.Linear(hidden_dim, 1)         # 預測 TP 百分比
        self.sl_head = nn.Linear(hidden_dim, 1)         # 預測 SL 百分比
        self.lev_head = nn.Linear(hidden_dim, 1)        # 槓桿預測（連續）

    def forward(self, x):
        shared_out = self.shared(x)
        logits = self.actor(shared_out)
        value = self.critic(shared_out)
        tp_out = self.tp_head(shared_out)
        sl_out = self.sl_head(shared_out)
        lev_out = self.lev_head(shared_out)
        return logits, value, tp_out, sl_out, lev_out

# ✅ 儲存模型
def save_model(model, path="a2c_model.pt"):
    torch.save(model.state_dict(), path)
    print(f"✅ 模型已儲存：{path}")

# ✅ 載入模型
def load_model_if_exists(model, path="a2c_model.pt"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"✅ 模型已載入：{path}")
    else:
        print("⚠️ 未找到 A2C 模型檔案，將使用未訓練的初始化模型")
