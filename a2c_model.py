import torch
import torch.nn as nn
import os

class ActorCritic(nn.Module):
    def __init__(self, input_dim: int = 33, hidden_dim: int = 64, action_dim: int = 2):  # input_dim=33
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

        # 初始化權重（例如 Xavier 初始化）
        self._init_weights()

    def forward(self, x):
        # 確保輸入是正確的形狀，展平為 (batch_size, input_dim)
        x = x.view(x.size(0), -1)
        shared_out = self.shared(x)
        logits = self.actor(shared_out)
        value = self.critic(shared_out)
        tp_out = self.tp_head(shared_out)
        sl_out = self.sl_head(shared_out)
        lev_out = self.lev_head(shared_out)

        return logits, value, tp_out, sl_out, lev_out

    def _init_weights(self):
        # 使用 Xavier 初始化權重
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

# 儲存模型
def save_model(model, path="a2c_model.pt"):
    try:
        torch.save(model.state_dict(), path)
        print(f"✅ 模型已儲存：{path}")
    except Exception as e:
        print(f"❌ 儲存模型時出錯：{e}")

# 載入模型
def load_model_if_exists(model, path="a2c_model.pt"):
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path))
            print(f"✅ 模型已載入：{path}")
        except Exception as e:
            print(f"❌ 載入模型時出錯：{e}")
    else:
        print("⚠️ 未找到 A2C 模型檔案，將使用未訓練的初始化模型")
