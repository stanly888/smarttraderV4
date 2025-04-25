# dqn_trainer.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from dqn_model import DQN, save_model, load_model_if_exists
from replay_buffer import ReplayBuffer
from reward_fetcher import get_real_reward

# 初始化模型與優化器
model = DQN(input_dim=35)  # ✅ 含 fib, bb, atr 的 35 維輸入
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Replay Buffer
buffer = ReplayBuffer(capacity=1000)
MODEL_PATH = "dqn_model.pt"
BUFFER_PATH = "dqn_replay.json"
load_model_if_exists(model, MODEL_PATH)
buffer.load(BUFFER_PATH)

TRAIN_STEPS = 20
BATCH_SIZE = 16

def simulate_reward(action: int, tp: float, sl: float, leverage: float, fib_distance: float) -> float:
    """模擬 reward，含 fib 距離懲罰項"""
    if action == 2:
        return np.random.uniform(-0.1, 0.2)
    raw = tp if np.random.rand() < 0.5 else -sl
    fee = 0.0004 * leverage * 2
    funding = 0.00025 * leverage
    base_reward = raw * leverage - fee - funding
    fib_penalty = abs(fib_distance - 0.618)
    return round(base_reward * (1 - fib_penalty), 4)

def train_dqn(features: np.ndarray) -> dict:
    atr = max(features[3], 0.002)
    fib_distance = features[16]
    bb_width = features[4]
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    total_reward = 0

    for _ in range(TRAIN_STEPS):
        direction_logits, tp_out, sl_out, lev_out = model(x)
        probs = torch.softmax(direction_logits, dim=-1)
        action = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, action].item()

        # ✅ TP/SL 統一邏輯：fib 越接近 0.618，配合 BB 寬度、ATR 決定區間
        fib_weight = max(1 - abs(fib_distance - 0.618), 0.2)
        tp = torch.sigmoid(tp_out).item() * bb_width * fib_weight * atr
        sl = max(torch.sigmoid(sl_out).item() * bb_width * fib_weight * atr, 0.002)
        leverage = torch.sigmoid(lev_out).item() * 9 + 1

        reward_val, _, _ = get_real_reward()
        if reward_val is None:
            reward_val = simulate_reward(action, tp, sl, leverage, fib_distance)

        total_reward += reward_val

        # ✅ 寫入 buffer
        buffer.add(
            state=x.squeeze(0).numpy(),
            action=action,
            reward=reward_val,
            next_state=x.squeeze(0).numpy(),
            done=False
        )

        # ✅ 訓練 DQN 網路
        if len(buffer) >= BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            if batch is None:
                continue
            try:
                states, actions, rewards, _, _ = batch
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
            except ValueError as e:
                print(f"[警告] DQN 回放失敗，state shape 不一致：{e}")
                continue

            direction_logits, _, _, _ = model(states)
            q_vals = direction_logits
            q_target = q_vals.clone().detach()
            for i in range(len(rewards)):
                q_target[i, actions[i]] = rewards[i]

            loss = F.mse_loss(q_vals, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 儲存模型與 Buffer
    save_model(model, MODEL_PATH)
    buffer.save(BUFFER_PATH)

    direction = "Long" if action == 0 else "Short" if action == 1 else "Skip"

    return {
        "model": "DQN",
        "direction": direction,
        "confidence": round(confidence, 4),
        "tp": round(tp, 4),
        "sl": round(sl, 4),
        "leverage": int(leverage),
        "score": round(total_reward / TRAIN_STEPS, 4),
        "fib_distance": round(fib_distance, 4)
    }