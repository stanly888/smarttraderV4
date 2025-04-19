# ppo_trainer.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ppo_model import UnifiedRLModel, save_model, load_model_if_exists

# 超參數
TRAIN_STEPS = 20
GAMMA = 0.99
LR = 1e-3

# 初始化模型與 optimizer
model = UnifiedRLModel(input_dim=10)
load_model_if_exists(model, "ppo_model.pt")
optimizer = optim.Adam(model.parameters(), lr=LR)

def simulate_reward(direction, confidence):
    """模擬 TP/SL 命中與方向信心的 reward"""
    if direction == "Long":
        tp_hit = np.random.rand() < 0.4 + 0.5 * confidence
        sl_hit = not tp_hit
    else:
        tp_hit = np.random.rand() < 0.3 + 0.4 * confidence
        sl_hit = not tp_hit

    reward = 1.0 if tp_hit else -1.0
    return reward, tp_hit, sl_hit

def train_ppo(features: np.ndarray) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    all_rewards = []

    for _ in range(TRAIN_STEPS):
        logits, value = model(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        reward_val, _, _ = simulate_reward("Long" if action.item() == 0 else "Short", probs[0, action].item())
        reward = torch.tensor([reward_val], dtype=torch.float32)
        all_rewards.append(reward.item())

        _, next_value = model(x)
        advantage = reward + GAMMA * next_value - value

        actor_loss = -dist.log_prob(action) * advantage.detach()
        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 輸出策略結果
    with torch.no_grad():
        logits, _ = model(x)
        probs = F.softmax(logits, dim=-1)
        confidence, action = torch.max(probs, dim=-1)

    direction = "Long" if action.item() == 0 else "Short"
    leverage = int(2 + 3 * confidence.item())  # 根據信心模擬槓桿
    tp = round(1.5 + 2 * confidence.item(), 2)  # 模擬 TP%
    sl = round(1.0 + 1 * (1 - confidence.item()), 2)  # 模擬 SL%
    score = np.mean(all_rewards)

    save_model(model, "ppo_model.pt")  # ✅ 儲存模型

    return {
        "model": "PPO",
        "direction": direction,
        "confidence": confidence.item(),
        "leverage": leverage,
        "tp": tp,
        "sl": sl,
        "score": score
    }
