# ppo_trainer.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ppo_model import UnifiedRLModel, save_model, load_model_if_exists
from replay_buffer import ReplayBuffer
from reward_fetcher import get_real_reward

TRAIN_STEPS = 20
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 8

model = UnifiedRLModel(input_dim=35)  # ✅ 升級為含 ATR/Fib/Price 共 35 維
load_model_if_exists(model, "ppo_model.pt")
optimizer = optim.Adam(model.parameters(), lr=LR)

replay_buffer = ReplayBuffer(capacity=1000)
replay_buffer.load("ppo_replay.json")

def simulate_reward(direction: str, tp: float, sl: float, leverage: float, fib_distance: float) -> float:
    hit = np.random.rand()
    raw = tp if hit < 0.5 else -sl
    fee = 0.0004 * leverage * 2
    funding = 0.00025 * leverage
    base_reward = raw * leverage - fee - funding
    fib_penalty = abs(fib_distance - 0.618)
    return round(base_reward * (1 - fib_penalty), 4)

def train_ppo(features: np.ndarray) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    atr = max(features[3], 0.002)
    fib_distance = features[16]

    logits, value, tp_out, sl_out, lev_out = model(x)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()

    direction = "Long" if action.item() == 0 else "Short"
    confidence = probs[0, action].item()
    tp = torch.sigmoid(tp_out).item() * atr
    sl = max(torch.sigmoid(sl_out).item() * atr, 0.002)
    leverage = torch.sigmoid(lev_out).item() * 9 + 1

    reward_val, _, _ = get_real_reward()
    if reward_val is None:
        reward_val = simulate_reward(direction, tp, sl, leverage, fib_distance)

    # ✅ 新 ReplayBuffer 結構：加入 next_state & done
    replay_buffer.add(
        state=x.squeeze(0).numpy(),
        action=action.item(),
        reward=reward_val,
        next_state=x.squeeze(0).numpy(),
        done=False
    )

    if len(replay_buffer) >= BATCH_SIZE:
        states, actions, rewards, _, _ = replay_buffer.sample(BATCH_SIZE)
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

    replay_buffer.save("ppo_replay.json")
    save_model(model, "ppo_model.pt")

    return {
        "model": "PPO",
        "direction": direction,
        "confidence": round(confidence, 4),
        "leverage": int(leverage),
        "tp": round(tp, 4),
        "sl": round(sl, 4),
        "score": round(reward_val, 4),
        "fib_distance": round(fib_distance, 4)
    }