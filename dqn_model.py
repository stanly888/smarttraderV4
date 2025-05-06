import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class DQN(nn.Module):
    def __init__(self, input_dim: int = 35, hidden_dim: int = 64):  # input_dim 改為 35
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 五個輸出：方向、止盈、止損、槓桿
        self.direction_head = nn.Linear(hidden_dim, 3)  # Long / Short / Skip
        self.tp_head = nn.Linear(hidden_dim, 1)         # TP %
        self.sl_head = nn.Linear(hidden_dim, 1)         # SL %
        self.lev_head = nn.Linear(hidden_dim, 1)        # Leverage 倍數

        # 初始化權重
        self._init_weights()

    def forward(self, x):
        # 確保輸入的形狀是正確的，展平輸入張量
        x = x.view(x.size(0), -1)  # 展平為 (batch_size, input_dim)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        direction_logits = self.direction_head(x)
        tp_out = self.tp_head(x)
        sl_out = self.sl_head(x)
        lev_out = self.lev_head(x)

        # 返回 Tensor 類型的輸出，不使用 .item()，保留梯度信息
        return direction_logits, tp_out, sl_out, lev_out

    def _init_weights(self):
        # 使用 Xavier 初始化權重
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

# 儲存模型
def save_model(model, path="dqn_model.pt"):
    try:
        torch.save(model.state_dict(), path)
        print(f"✅ DQN 模型已儲存：{path}")
    except Exception as e:
        print(f"❌ 儲存模型時出錯：{e}")

# 載入模型
def load_model_if_exists(model, path="dqn_model.pt"):
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path))
            print(f"✅ DQN 模型已載入：{path}")
        except Exception as e:
            print(f"❌ 載入模型時出錯：{e}")
    else:
        print("⚠️ 未找到 DQN 模型檔案，使用未訓練參數")
