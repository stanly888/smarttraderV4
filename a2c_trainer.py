# a2c_trainer.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from a2c_model import ActorCritic
from replay_buffer import ReplayBuffer

model = ActorCritic(input_dim=10, action_dim=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
TRAIN_STEPS = 20
GAMMA = 0.99
replay_buffer = ReplayBuffer(capacity=1000)

def simulate_reward(direction: str, tp: float, sl: float, leverage: float) -> float:
    """模擬 TP/SL 命中 + 槓桿後 reward，含手續費與資金費"""
    if np.random.rand() < 0.5:
        raw = tp
    else:
        raw = -sl
    fee = 0.0004 * leverage * 2      # 假設總手續費 0.08%
    funding = 0.00025 * leverage     # 資金費率
    return raw * leverage - fee - funding

def train_a2c(features: np.ndarray) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    total_reward = 0

    for _ in range(TRAIN_STEPS):
        logits, value, tp_out, sl_out, lev_out = model(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        direction = "Long" if action.item() == 0 else "Short"
        confidence = probs[0, action].item()
        tp = torch.sigmoid(tp_out).item() * 3.5
        sl = torch.sigmoid(sl_out).item() * 2.0
        leverage = torch.sigmoid(lev_out).item() * 9 + 1

        reward = simulate_reward(direction, tp, sl, leverage)
        total_reward += reward

        replay_buffer.push(x.squeeze(0).numpy(), action.item(), reward)

        _, next_value, _, _, _ = model(x)
        advantage = torch.tensor([reward], dtype=torch.float32) + GAMMA * next_value - value

        actor_loss = -dist.log_prob(action) * advantage.detach()
        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Replay Buffer 回放訓練
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

    # 輸出策略
    with torch.no_grad():
        logits, _, tp_out, sl_out, lev_out = model(x)
        probs = F.softmax(logits, dim=-1)
        confidence, selected = torch.max(probs, dim=-1)
        tp = torch.sigmoid(tp_out).item() * 3.5
        sl = torch.sigmoid(sl_out).item() * 2.0
        leverage = torch.sigmoid(lev_out).item() * 9 + 1

    return {
        "model": "A2C",
        "direction": "Long" if selected.item() == 0 else "Short",
        "confidence": round(confidence.item(), 4),
        "tp": round(tp, 2),
        "sl": round(sl, 2),
        "leverage": int(leverage),
        "score": round(total_reward / TRAIN_STEPS, 4)
    }