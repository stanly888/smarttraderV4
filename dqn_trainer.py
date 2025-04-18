# dqn_trainer.py
import torch
import torch.optim as optim
import numpy as np
from dqn_model import DQNModel

model = DQNModel(input_dim=10)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
gamma = 0.95

def train_dqn(features: np.ndarray) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    q_values = model(x)
    action = torch.argmax(q_values, dim=1).item()

    # 模擬 reward
    reward = torch.tensor([np.random.uniform(-1, 1)], dtype=torch.float32)

    # 模擬 next state（假設無重大改變）
    next_q_values = model(x).detach()
    max_next_q = next_q_values.max().item()

    target_q = reward + gamma * max_next_q
    predicted_q = q_values[0, action]

    loss = (predicted_q - target_q).pow(2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    direction = "Long" if action == 0 else "Short"
    confidence = float(torch.softmax(q_values, dim=1)[0, action].item())
    tp = round(1.5 + confidence * 2.5, 2)
    sl = round(1.0 + (1 - confidence) * 2.0, 2)
    leverage = int(min(10, max(1, int(confidence * 12))))

    return {
        "model": "DQN",
        "direction": direction,
        "confidence": confidence,
        "tp": tp,
        "sl": sl,
        "leverage": leverage,
        "score": float(reward.item())
    }
