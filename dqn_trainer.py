# dqn_trainer.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from dqn_model import DQN

# 初始化模型與優化器
model = DQN(input_dim=10)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 模擬 reward 機制
def simulate_reward(action: int) -> float:
    """
    模擬 reward，未來應替換為真實 TP/SL 結果
    """
    if action == 0:  # Long
        return np.random.uniform(-1.0, 1.5)
    elif action == 1:  # Short
        return np.random.uniform(-1.0, 1.5)
    else:  # Skip
        return np.random.uniform(-0.1, 0.2)

# 訓練次數（每次 retrain）
TRAIN_STEPS = 20

def train_dqn(features: np.ndarray) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    total_reward = 0

    for _ in range(TRAIN_STEPS):
        q_values = model(x)
        action = torch.argmax(q_values, dim=-1).item()

        reward = simulate_reward(action)
        total_reward += reward

        target_q = q_values.clone().detach()
        target_q[0, action] = reward  # target only更新選定動作

        loss = F.mse_loss(q_values, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_reward = total_reward / TRAIN_STEPS
    confidence = torch.softmax(model(x), dim=-1)[0, action].item()

    return {
        "model": "DQN",
        "direction": "Long" if action == 0 else "Short" if action == 1 else "Skip",
        "confidence": round(confidence, 3),
        "tp": round(np.random.uniform(1.0, 3.5), 2),
        "sl": round(np.random.uniform(1.0, 2.5), 2),
        "leverage": np.random.choice([2, 3, 5]),
        "score": round(avg_reward, 4)
    }
