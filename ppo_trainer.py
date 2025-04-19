# ppo_trainer.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ppo_model import UnifiedRLModel, save_model, load_model_if_exists
from replay_buffer import ReplayBuffer  # ✅ 引入 replay buffer

# 超參數
TRAIN_STEPS = 20
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 8

# 初始化模型與 optimizer
model = UnifiedRLModel(input_dim=10)
load_model_if_exists(model, "ppo_model.pt")
optimizer = optim.Adam(model.parameters(), lr=LR)
replay_buffer = ReplayBuffer(capacity=1000)  # ✅ 建立 replay buffer

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

    # Step 1: 模擬動作與 reward
    logits, value = model(x)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    direction = "Long" if action.item() == 0 else "Short"
    confidence = probs[0, action].item()

    reward_val, _, _ = simulate_reward(direction, confidence)
    reward = torch.tensor([reward_val], dtype=torch.float32)
    replay_buffer.push(x.squeeze(0).numpy(), action.item(), reward_val)

    # Step 2: 訓練階段（使用 replay buffer）
    if len(replay_buffer) >= BATCH_SIZE:
        states, actions, rewards = replay_buffer.sample(BATCH_SIZE)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        for _ in range(TRAIN_STEPS):
            logits, values = model(states)
            dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
            log_probs = dist.log_prob(actions)
            _, next_values = model(states)
            advantages = rewards + GAMMA * next_values.squeeze() - values.squeeze()

            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = advantages.pow(2).mean()
            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Step 3: 輸出策略結果
    with torch.no_grad():
        logits, _ = model(x)
        probs = F.softmax(logits, dim=-1)
        confidence, action = torch.max(probs, dim=-1)

    direction = "Long" if action.item() == 0 else "Short"
    leverage = int(2 + 3 * confidence.item())
    tp = round(1.5 + 2 * confidence.item(), 2)
    sl = round(1.0 + 1 * (1 - confidence.item()), 2)

    save_model(model, "ppo_model.pt")

    return {
        "model": "PPO",
        "direction": direction,
        "confidence": confidence.item(),
        "leverage": leverage,
        "tp": tp,
        "sl": sl,
        "score": reward_val
    }
