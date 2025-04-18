import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from a2c_model import ActorCritic

model = ActorCritic(input_dim=10)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

TRAIN_STEPS = 20

def simulate_reward(direction: str, tp: float, sl: float) -> float:
    """
    模擬獎勵，方向正確 + 命中 TP 給正報酬，否則負報酬。
    可換成實際交易結果回饋。
    """
    success = np.random.rand()
    if direction == "Long":
        return tp if success > 0.4 else -sl
    elif direction == "Short":
        return tp if success > 0.4 else -sl
    else:
        return -0.1

def train_a2c(features: np.ndarray) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    total_reward = 0

    for _ in range(TRAIN_STEPS):
        logits, value = model(x)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1)

        direction = ["Hold", "Long", "Short"][action.item()]
        tp = round(np.random.uniform(1.5, 3.5), 2)
        sl = round(np.random.uniform(1.0, 2.0), 2)

        reward = simulate_reward(direction, tp, sl)
        total_reward += reward

        _, next_value = model(x)
        advantage = torch.tensor([reward]) + 0.99 * next_value - value

        actor_loss = -torch.log(probs[0, action]) * advantage.detach()
        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    confidence = float(probs[0, action])
    return {
        "model": "A2C",
        "direction": direction,
        "confidence": round(confidence, 4),
        "tp": tp,
        "sl": sl,
        "leverage": np.random.choice([3, 5, 10]),
        "score": round(total_reward / TRAIN_STEPS, 4)
    }
