# a2c_trainer.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from a2c_model import ActorCritic
from replay_buffer import ReplayBuffer
from reward_fetcher import get_real_reward

model = ActorCritic(input_dim=35, action_dim=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
TRAIN_STEPS = 20
GAMMA = 0.99

replay_buffer = ReplayBuffer(capacity=1000)
replay_buffer.load("a2c_replay.json")

def simulate_reward(direction: str, tp: float, sl: float, leverage: float, fib_distance: float) -> float:
    raw = tp if np.random.rand() < 0.5 else -sl
    fee = 0.0004 * leverage * 2
    funding = 0.00025 * leverage
    base = raw * leverage - fee - funding
    fib_penalty = abs(fib_distance - 0.618)
    return round(base * (1 - fib_penalty), 4)

def train_a2c(features: np.ndarray) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    atr = max(features[3], 0.002)
    fib_distance = features[16]
    bb_width = features[4]

    total_reward = 0

    for _ in range(TRAIN_STEPS):
        logits, value, tp_out, sl_out, lev_out = model(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        direction = "Long" if action.item() == 0 else "Short"
        confidence = probs[0, action].item()

        fib_weight = max(1 - abs(fib_distance - 0.618), 0.2)
        tp = torch.sigmoid(tp_out).item() * bb_width * fib_weight * atr
        sl = max(torch.sigmoid(sl_out).item() * bb_width * fib_weight * atr, 0.002)
        leverage = torch.sigmoid(lev_out).item() * 9 + 1

        reward_val, _, _ = get_real_reward()
        if reward_val is None:
            reward_val = simulate_reward(direction, tp, sl, leverage, fib_distance)

        total_reward += reward_val

        replay_buffer.add(
            state=x.squeeze(0).numpy(),
            action=action.item(),
            reward=reward_val,
            next_state=x.squeeze(0).numpy(),
            done=False
        )

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
            if batch is None:
                continue
            states, actions, rewards, _, _ = batch
            for state, action, reward in zip(states, actions, rewards):
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

        fib_weight = max(1 - abs(fib_distance - 0.618), 0.2)
        tp = torch.sigmoid(tp_out).item() * bb_width * fib_weight * atr
        sl = max(torch.sigmoid(sl_out).item() * bb_width * fib_weight * atr, 0.002)
        leverage = torch.sigmoid(lev_out).item() * 9 + 1

    return {
        "model": "A2C",
        "direction": "Long" if selected.item() == 0 else "Short",
        "confidence": round(confidence.item(), 4),
        "tp": round(tp, 4),
        "sl": round(sl, 4),
        "leverage": int(leverage),
        "score": round(total_reward / TRAIN_STEPS, 4),
        "fib_distance": round(fib_distance, 4)
    }