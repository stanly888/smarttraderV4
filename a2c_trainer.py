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

def simulate_reward(direction: str, tp: float, sl: float) -> float:
    success = np.random.rand()
    return tp if success > 0.45 else -sl

def train_a2c(features: np.ndarray) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    total_reward = 0

    for _ in range(TRAIN_STEPS):
        logits, value = model(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        direction = "Long" if action.item() == 0 else "Short"
        tp = round(np.random.uniform(1.5, 3.0), 2)
        sl = round(np.random.uniform(1.0, 2.0), 2)
        reward = simulate_reward(direction, tp, sl)
        total_reward += reward

        replay_buffer.push(x, action, reward)

        _, next_value = model(x)
        advantage = torch.tensor([reward], dtype=torch.float32) + GAMMA * next_value - value

        actor_loss = -dist.log_prob(action) * advantage.detach()
        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Replay Buffer 經驗回放
    if len(replay_buffer) >= 5:
        for _ in range(3):  # 回放次數
            batch = replay_buffer.sample(5)
            for state, action, reward in zip(*batch[:3]):
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action_tensor = torch.tensor(action, dtype=torch.int64)  # ✅ 修正關鍵行

                logits, value = model(state_tensor)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                _, next_value = model(state_tensor)
                advantage = torch.tensor([reward], dtype=torch.float32) + GAMMA * next_value - value

                actor_loss = -dist.log_prob(action_tensor) * advantage.detach()
                critic_loss = advantage.pow(2)
                loss = actor_loss + critic_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    with torch.no_grad():
        logits, _ = model(x)
        probs = F.softmax(logits, dim=-1)
        confidence, selected = torch.max(probs, dim=-1)

    return {
        "model": "A2C",
        "direction": "Long" if selected.item() == 0 else "Short",
        "confidence": round(confidence.item(), 4),
        "tp": tp,
        "sl": sl,
        "leverage": int(2 + 3 * confidence.item()),
        "score": round(total_reward / TRAIN_STEPS, 4)
    }
