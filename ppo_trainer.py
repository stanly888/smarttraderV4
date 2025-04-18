# ppo_trainer.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ppo_model import PPOActorCritic

model = PPOActorCritic(input_dim=10)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
eps_clip = 0.2
gamma = 0.99

def train_ppo(features: np.ndarray) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    probs_old, value_old = model(x).detach()
    action = torch.multinomial(probs_old, 1)

    # 模擬 reward
    reward = torch.tensor([np.random.uniform(-1, 1)], dtype=torch.float32)

    for _ in range(20):  # 多次更新
        probs, value = model(x)
        dist_ratio = (probs[0, action] / (probs_old[0, action] + 1e-8)).clamp(1 - eps_clip, 1 + eps_clip)
        advantage = reward + gamma * value - value_old

        actor_loss = -dist_ratio * advantage.detach()
        critic_loss = F.mse_loss(value, reward + gamma * value_old)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

    # 預測結果
    with torch.no_grad():
        probs, _ = model(x)
        direction = "Long" if probs[0][0] > probs[0][1] else "Short"
        confidence = float(probs.max().item())

    tp = round(1.8 + confidence * 2.5, 2)
    sl = round(1.2 + (1 - confidence) * 2.0, 2)
    leverage = int(min(10, max(1, int(confidence * 10))))

    return {
        "model": "PPO",
        "direction": direction,
        "confidence": confidence,
        "tp": tp,
        "sl": sl,
        "leverage": leverage,
        "score": float(reward.item())
    }
