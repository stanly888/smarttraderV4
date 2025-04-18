# a2c_trainer.py
import torch
import torch.optim as optim
import numpy as np
from a2c_model import ActorCritic

# 初始化
model = ActorCritic(input_dim=10)  # 你的技術指標輸入維度
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# 訓練次數（每回 retrain）
TRAIN_STEPS = 20

def train_a2c(features: np.ndarray) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    all_rewards = []

    # 模擬環境回饋，這裡暫時用隨機模擬 reward，可換成真實交易 log
    for _ in range(TRAIN_STEPS):
        logits, value = model(x)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1)

        # 模擬 reward，未來可由實際交易結果生成
        reward = torch.tensor([np.random.uniform(-1, 1)], dtype=torch.float32)
        all_rewards.append(reward.item())

        _, next_value = model(x)
        advantage = reward + 0.99 * next_value - value

        actor_loss = -torch.log(probs[0, action]) * advantage.detach()
        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 輸出方向與信心
    with torch.no_grad():
        logits, _ = model(x)
        probs = F.softmax(logits, dim=-1)
        direction = "Long" if probs[0][0] > probs[0][1] else "Short"
        confidence = float(probs.max().item())

    # 模擬 TP/SL 與槓桿（可改為訓練預測）
    tp = round(1.5 + confidence * 3, 2)
    sl = round(1.0 + (1 - confidence) * 2, 2)
    leverage = int(min(5, max(1, int(confidence * 10))))

    return {
        "model": "A2C",
        "direction": direction,
        "confidence": confidence,
        "tp": tp,
        "sl": sl,
        "leverage": leverage,
        "score": np.mean(all_rewards)  # 可作為選模型依據
    }
