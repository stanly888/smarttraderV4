import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from a2c_model import ActorCritic
from replay_buffer import ReplayBuffer
from reward_fetcher import get_real_reward
from reward_utils import simulate_reward  # ✅ 外部載入 simulate_reward

MODEL_PATH = "a2c_model.pt"
REPLAY_PATH = "a2c_replay.json"
TRAIN_STEPS = 20
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 5

model = ActorCritic(input_dim=35, action_dim=2)
optimizer = optim.Adam(model.parameters(), lr=LR)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print("✅ A2C 模型已載入")

replay_buffer = ReplayBuffer(capacity=1000)
replay_buffer.load(REPLAY_PATH)
if len(replay_buffer) > 0:
    print("✅ A2C Replay Buffer 已載入")

def train_a2c(features: np.ndarray, atr: float, bb_width: float, fib_distance: float) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
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
        advantage = torch.tensor([reward_val], dtype=torch.float32) + GAMMA * next_value.detach() - value

        actor_loss = -dist.log_prob(action) * advantage.detach()
        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if len(replay_buffer) >= BATCH_SIZE:
        for _ in range(3):
            try:
                states, actions, rewards, _, _ = replay_buffer.sample(BATCH_SIZE)
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64)
                rewards = torch.tensor(rewards, dtype=torch.float32)

                logits, values, _, _, _ = model(states)
                dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
                log_probs = dist.log_prob(actions)
                _, next_values, _, _, _ = model(states)

                advantages = rewards + GAMMA * next_values.squeeze().detach() - values.squeeze()
                actor_loss = -(log_probs * advantages.detach()).mean()
                critic_loss = advantages.pow(2).mean()
                loss = actor_loss + critic_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"⚠️ A2C 回放訓練失敗：{e}")

    torch.save(model.state_dict(), MODEL_PATH)
    replay_buffer.save(REPLAY_PATH)

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
        "tp": round(tp * 100, 2),
        "sl": round(sl * 100, 2),
        "leverage": int(leverage),
        "score": round(total_reward / TRAIN_STEPS, 4),
        "fib_distance": round(fib_distance, 4)
    }