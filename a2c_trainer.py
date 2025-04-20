# a2c_trainer.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from a2c_model import ActorCritic
from replay_buffer import ReplayBuffer
from reward_fetcher import get_real_reward

model = ActorCritic(input_dim=35, action_dim=2)  # ✅ 升級為 34 維輸入
optimizer = optim.Adam(model.parameters(), lr=1e-3)
TRAIN_STEPS = 20
GAMMA = 0.99

replay_buffer = ReplayBuffer(capacity=1000)
replay_buffer.load("a2c_replay.json")

def simulate_reward(direction: str, tp: float, sl: float, leverage: float) -> float:
    raw = tp if np.random.rand() < 0.5 else -sl
    fee = 0.0004 * leverage * 2
    funding = 0.00025 * leverage
    return raw * leverage - fee - funding

def train_a2c(features: np.ndarray) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    atr = max(features[3], 0.002)  # ✅ 從第 4 維（未標準化 ATR）擷取 ATR 值

    total_reward = 0

    for _ in range(TRAIN_STEPS):
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
            reward_val = simulate_reward(direction, tp, sl, leverage)

        total_reward += reward_val
        replay_buffer.push(x.squeeze(0).numpy(), action.item(), reward_val)

        _, next_value, _, _, _ = model(x)
        advantage = torch.tensor([reward_val], dtype=torch.float32) + GAMMA * next_value - value

        actor_loss = -dist.log_prob(action) * advantage.detach()
        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if len(replay_buffer) >= 5:
        for _ in range(3):
            batch = replay_buffer.sample(5)
            for state, action, reward in zip(*batch[:3]):
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action_tensor = torch.tensor(action, dtype=torch.int64)

                logits, value, _, _, _ = model(state_tensor)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                _, next_value, _, _, _ = model(state_tensor)
                advantage = torch.tensor([reward], dtype=torch.float32) + GAMMA * next_value - value

                actor_loss = -dist.log_prob(action_tensor) * advantage.detach()
                critic_loss = advantage.pow(2)
                loss = actor_loss + critic_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    replay_buffer.save("a2c_replay.json")

    with torch.no_grad():
        logits, _, tp_out, sl_out, lev_out = model(x)
        probs = F.softmax(logits, dim=-1)
        confidence, selected = torch.max(probs, dim=-1)
        tp = torch.sigmoid(tp_out).item() * atr
        sl = max(torch.sigmoid(sl_out).item() * atr, 0.002)
        leverage = torch.sigmoid(lev_out).item() * 9 + 1

    return {
        "model": "A2C",
        "direction": "Long" if selected.item() == 0 else "Short",
        "confidence": round(confidence.item(), 4),
        "tp": round(tp, 4),
        "sl": round(sl, 4),
        "leverage": int(leverage),
        "score": round(total_reward / TRAIN_STEPS, 4)
    }