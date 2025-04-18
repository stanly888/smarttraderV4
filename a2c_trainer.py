# a2c_trainer.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from a2c_model import ActorCritic

# 初始化模型與訓練設定
model = ActorCritic(input_dim=10)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
TRAIN_STEPS = 20

def train_a2c(features: np.ndarray) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    total_reward = 0

    for _ in range(TRAIN_STEPS):
        logits, value = model(x)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1).item()

        # 模擬 reward（未來可替換成真實交易獎勵）
        reward = np.random.uniform(-1, 1)
        total_reward += reward

        # 下一步評估與損失計算
        _, next_value = model(x)
        advantage = torch.tensor([reward + 0.99 * next_value.item() - value.item()])
        actor_loss = -torch.log(probs[0, action]) * advantage.detach()
        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 模擬產出（未來可改為模型推理）
    direction = "Long" if action == 1 else "Short"
    confidence = float(probs[0, action].item())
    tp = round(np.random.uniform(1, 4), 2)
    sl = round(np.random.uniform(0.8, 2), 2)
    leverage = np.random.choice([1, 2, 3, 4, 5])
    score = round(total_reward / TRAIN_STEPS, 4)

    return {
        "model": "A2C",
        "direction": direction,
        "confidence": confidence,
        "tp": tp,
        "sl": sl,
        "leverage": leverage,
        "score": score
    }
