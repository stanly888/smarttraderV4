# dqn_trainer.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from dqn_model import DQN
from replay_buffer import ReplayBuffer

# 初始化
model = DQN(input_dim=10)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
buffer = ReplayBuffer(capacity=1000)

# 模型儲存與載入
MODEL_PATH = "dqn_model.pt"

def save_model(model, path=MODEL_PATH):
    torch.save(model.state_dict(), path)
    print(f"✅ DQN 模型已儲存：{path}")

def load_model_if_exists(model, path=MODEL_PATH):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"✅ DQN 模型已載入：{path}")
    else:
        print("⚠️ 未找到 DQN 模型檔案，使用未訓練參數")

load_model_if_exists(model)

# 訓練次數
TRAIN_STEPS = 20

def simulate_reward(action: int) -> float:
    if action == 0: return np.random.uniform(-1.0, 1.5)  # Long
    elif action == 1: return np.random.uniform(-1.0, 1.5)  # Short
    else: return np.random.uniform(-0.1, 0.2)  # Skip

def train_dqn(features: np.ndarray) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    total_reward = 0

    for _ in range(TRAIN_STEPS):
        q_values = model(x)
        action = torch.argmax(q_values, dim=-1).item()
        reward = simulate_reward(action)
        total_reward += reward

        # 儲存到 Replay Buffer
        buffer.push(x.squeeze(0).numpy(), action, reward)

        # 隨機取樣訓練（回放記憶）
        if len(buffer) > 16:
            batch = buffer.sample(16)
            states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
            actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
            rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)

            q_vals = model(states)
            q_target = q_vals.clone().detach()
            for i in range(16):
                q_target[i, actions[i]] = rewards[i]

            loss = F.mse_loss(q_vals, q_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 輸出結果
    avg_reward = total_reward / TRAIN_STEPS
    confidence = torch.softmax(model(x), dim=-1)[0, action].item()

    save_model(model)

    return {
        "model": "DQN",
        "direction": "Long" if action == 0 else "Short" if action == 1 else "Skip",
        "confidence": round(confidence, 3),
        "tp": round(np.random.uniform(1.0, 3.5), 2),
        "sl": round(np.random.uniform(1.0, 2.5), 2),
        "leverage": np.random.choice([2, 3, 5]),
        "score": round(avg_reward, 4)
    }
