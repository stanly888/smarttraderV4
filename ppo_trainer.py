import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from ppo_model import UnifiedRLModel, save_model, load_model_if_exists
from replay_buffer import ReplayBuffer
from reward_fetcher import get_real_reward
from reward_utils import simulate_reward  # ✅ 引入統一的 simulate_reward

TRAIN_STEPS = 20
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 8

MODEL_PATH = "ppo_model.pt"
REPLAY_PATH = "ppo_replay.json"

model = UnifiedRLModel(input_dim=35)
load_model_if_exists(model, MODEL_PATH)
optimizer = optim.Adam(model.parameters(), lr=LR)

replay_buffer = ReplayBuffer(capacity=1000)
replay_buffer.load(REPLAY_PATH)
if len(replay_buffer) > 0:
    print("✅ PPO Replay Buffer 已載入")

def train_ppo(features: np.ndarray, atr: float, bb_width: float, fib_distance: float) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

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
        reward_val = simulate_reward(
            direction=direction,
            tp=tp,
            sl=sl,
            leverage=leverage,
            fib_distance=fib_distance
        )

    replay_buffer.add(
        state=x.squeeze(0).numpy(),
        action=action.item(),
        reward=reward_val,
        next_state=x.squeeze(0).numpy(),
        done=False
    )

    if len(replay_buffer) >= BATCH_SIZE:
        try:
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
        except Exception as e:
            print(f"⚠️ PPO 回放訓練失敗：{e}")

    replay_buffer.save(REPLAY_PATH)
    save_model(model, MODEL_PATH)

    return {
        "model": "PPO",
        "direction": direction,
        "confidence": round(confidence, 4),
        "leverage": int(leverage),
        "tp": round(tp * 100, 2),  # ✅ 同步顯示成百分比
        "sl": round(sl * 100, 2),
        "score": round(reward_val, 4),
        "fib_distance": round(fib_distance, 4)
    }