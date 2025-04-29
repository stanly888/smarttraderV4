import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class DQN(nn.Module):
    def __init__(self, input_dim: int = 35, hidden_dim: int = 64):  # ✅ input_dim 改為 35
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 五個輸出：方向、止盈、止損、槓桿
        self.direction_head = nn.Linear(hidden_dim, 3)  # Long / Short / Skip
        self.tp_head = nn.Linear(hidden_dim, 1)         # TP %
        self.sl_head = nn.Linear(hidden_dim, 1)         # SL %
        self.lev_head = nn.Linear(hidden_dim, 1)        # Leverage 倍數

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        direction_logits = self.direction_head(x)
        tp_out = self.tp_head(x)
        sl_out = self.sl_head(x)
        lev_out = self.lev_head(x)

        # 在這裡將 Tensor 類型數值使用 .item() 轉換為純數字 (float)
        return direction_logits, tp_out.item(), sl_out.item(), lev_out.item()

# ✅ 儲存模型
def save_model(model, path="dqn_model.pt"):
    torch.save(model.state_dict(), path)
    print(f"✅ DQN 模型已儲存：{path}")

# ✅ 載入模型
def load_model_if_exists(model, path="dqn_model.pt"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"✅ DQN 模型已載入：{path}")
    else:
        print("⚠️ 未找到 DQN 模型檔案，使用未訓練參數")