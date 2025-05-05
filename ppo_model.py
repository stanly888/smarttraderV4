import torch
import torch.nn as nn
import os

class UnifiedRLModel(nn.Module):
    def __init__(self, input_dim=35, hidden_dim=64):
        super(UnifiedRLModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # 五輸出
        self.policy_head = nn.Linear(hidden_dim, 2)   # direction: Long / Short
        self.value_head = nn.Linear(hidden_dim, 1)    # critic value
        self.tp_head = nn.Linear(hidden_dim, 1)       # take profit %
        self.sl_head = nn.Linear(hidden_dim, 1)       # stop loss %
        self.lev_head = nn.Linear(hidden_dim, 1)      # leverage 倍數

        # 初始化模型權重
        self._init_weights()

    def forward(self, x):
        x = self.shared(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        tp = self.tp_head(x)
        sl = self.sl_head(x)
        lev = self.lev_head(x)

        # 返回數值保持為 Tensor 類型，以便進行反向傳播
        return logits, value, tp, sl, lev

    def _init_weights(self):
        """使用 Xavier 初始化方法初始化權重"""
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

# 儲存模型
def save_model(model, path="ppo_model.pt"):
    torch.save(model.state_dict(), path)
    print(f"✅ PPO 模型已儲存：{path}")

# 載入模型
def load_model_if_exists(model, path="ppo_model.pt"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"✅ PPO 模型已載入：{path}")
    else:
        print("⚠️ 未找到 PPO 模型檔案，使用未訓練參數")