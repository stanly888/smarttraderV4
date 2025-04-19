# dqn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class DQN(nn.Module):
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, output_dim: int = 3):
        """
        output_dim 預設為 3，對應 [Long, Short, Skip]
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

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
