# dqn_trainer.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from dqn_model import DQN, save_model, load_model_if_exists
from replay_buffer import ReplayBuffer
from reward_fetcher import get_real_reward

model = DQN(input_dim=33)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
buffer = ReplayBuffer(capacity=1000)

MODEL_PATH = "dqn_model.pt"
BUFFER_PATH = "dqn_replay.json"

load_model_if_exists(model, MODEL_PATH)
buffer.load(BUFFER_PATH)

TRAIN_STEPS = 20

def simulate_reward(action: int, tp: float, sl: float, leverage: float) -> float:
    if action == 2:
        return np.random.uniform(-0.1, 0.2)
    hit = np.random.rand()
    raw = tp if hit < 0.5 else -sl
    fee = 0.0004 * leverage * 2
    funding = 0.00025 * leverage
    return raw * leverage - fee - funding

def train_dqn(features: np.ndarray) -> dict:
    # ✅ 自動使用 features[3] 的 ATR 值（加保底避免為 0）
    atr = max(features[3], 0.002)
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    total_reward = 0

    for _ in range(TRAIN_STEPS):
        direction_logits, tp_out, sl_out, lev_out = model(x)
        probs = torch.softmax(direction_logits, dim=-1)
        action = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, action].item()

        # ✅ 模型預測比例 × ATR
        tp = torch.sigmoid(tp_out).item() * atr
        sl = max(torch.sigmoid(sl_out).item() * atr, 0.002)
        leverage = torch.sigmoid(lev_out).item() * 9 + 1

        reward_val, _, _ = get_real_reward()
        if reward_val is None:
            reward_val = simulate_reward(action, tp, sl, leverage)

        total_reward += reward_val
        buffer.push(x.squeeze(0).numpy(), action, reward_val)

        if len(buffer) > 16:
            batch = buffer.sample(16)
            try:
                states = torch.tensor(np.stack([np.array(b[0]).flatten() for b in batch]), dtype=torch.float32)
            except ValueError as e:
                print(f"[警告] DQN 回放失敗，state shape 不一致：{e}")
                continue

            actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
            rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)

            direction_logits, _, _, _ = model(states)
            q_vals = direction_logits
            q_target = q_vals.clone().detach()
            for i in range(len(batch)):
                q_target[i, actions[i]] = rewards[i]

            loss = F.mse_loss(q_vals, q_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    save_model(model, MODEL_PATH)
    buffer.save(BUFFER_PATH)

    direction = "Long" if action == 0 else "Short" if action == 1 else "Skip"

    return {
        "model": "DQN",
        "direction": direction,
        "confidence": round(confidence, 3),
        "tp": round(tp * 100, 2),
        "sl": round(sl * 100, 2),
        "leverage": int(leverage),
        "score": round(total_reward / TRAIN_STEPS, 4)
    }