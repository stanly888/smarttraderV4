# ppo_trainer.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ppo_model import UnifiedRLModel, save_model, load_model_if_exists
from replay_buffer import ReplayBuffer
from reward_fetcher import get_real_reward  # ✅ 實盤 reward 模組

# 超參數
TRAIN_STEPS = 20
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 8

# 初始化模型與 optimizer
model = UnifiedRLModel(input_dim=33)
load_model_if_exists(model, "ppo_model.pt")
optimizer = optim.Adam(model.parameters(), lr=LR)

# ✅ 建立並載入 Replay Buffer 記憶
replay_buffer = ReplayBuffer(capacity=1000)
replay_buffer.load("ppo_replay.json")

def simulate_reward(direction: str, tp: float, sl: float, leverage: float) -> float:
    hit = np.random.rand()
    raw_profit = tp if hit < 0.5 else -sl
    fee = 0.0004 * leverage * 2
    funding = 0.00025 * leverage
    return round(raw_profit * leverage - fee - funding, 4)

def train_ppo(features: np.ndarray) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    logits, value, tp_out, sl_out, lev_out = model(x)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()

    direction = "Long" if action.item() == 0 else "Short"
    confidence = probs[0, action].item()
    tp = torch.sigmoid(tp_out).item() * 3.5
    sl = torch.sigmoid(sl_out).item() * 2.0
    leverage = torch.sigmoid(lev_out).item() * 9 + 1

    reward_val, hit_tp, hit_sl = get_real_reward()
    if reward_val is None:
        reward_val = simulate_reward(direction, tp, sl, leverage)

    reward = torch.tensor([reward_val], dtype=torch.float32)
    replay_buffer.push(x.squeeze(0).numpy(), action.item(), reward_val)

    if len(replay_buffer) >= BATCH_SIZE:
        states, actions, rewards = replay_buffer.sample(BATCH_SIZE)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        for _ in range(TRAIN_STEPS):
            logits, values, _, _, _ = model(states)
            dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
            log_probs = dist.log_prob(actions)
            _, next_values, _, _, _ = model(states)

            advantages = rewards + GAMMA * next_values.squeeze() - values.squeeze()
            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = advantages.pow(2).mean()
            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ✅ 儲存 Replay Buffer
    replay_buffer.save("ppo_replay.json")
    save_model(model, "ppo_model.pt")

    return {
        "model": "PPO",
        "direction": direction,
        "confidence": round(confidence, 4),
        "leverage": int(leverage),
        "tp": round(tp, 2),
        "sl": round(sl, 2),
        "score": round(reward_val, 4)
    }