import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from a2c_model import ActorCritic
from replay_buffer import ReplayBuffer
from reward_fetcher import get_real_reward
from reward_utils import simulate_reward  # ✅ 外部引入 simulate_reward 函數

MODEL_PATH = "a2c_model.pt"
REPLAY_PATH = "a2c_replay.json"
TRAIN_STEPS = 20
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 5

# 初始化模型和優化器
model = ActorCritic(input_dim=35, action_dim=2)
optimizer = optim.Adam(model.parameters(), lr=LR)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print("✅ A2C 模型已載入")

# 初始化回放緩衝區
replay_buffer = ReplayBuffer(capacity=1000)
replay_buffer.load(REPLAY_PATH)
if len(replay_buffer) > 0:
    print("✅ A2C 回放緩衝區已載入")

def train_a2c(features: np.ndarray, atr: float, bb_width: float, fib_distance: float, confidence_threshold: float = 0.7) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # 將特徵轉換為tensor
    total_reward = 0

    for _ in range(TRAIN_STEPS):
        # 進行前向傳播，得到模型的輸出
        logits, value, tp_out, sl_out, lev_out = model(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()  # 從概率分佈中抽取一個行動

        direction = "Long" if action.item() == 0 else "Short"
        confidence = probs[0, action].item()

        # ✅ 信心過濾：信心低的情況直接選擇 Skip（目前A2C二選一，保留此功能）
        if confidence < confidence_threshold:
            # 這裡如果想更細可以自己加判斷，比如減小倉位，現在先保留直接進場但標註信心低。

            pass  # 目前保留（A2C二分類，不進行強制改動）

        # ✅ TP/SL動態調整
        fib_weight = max(1 - abs(fib_distance - 0.618), 0.2)  # 根據斐波那契進行加權
        tp = torch.sigmoid(tp_out).item() * bb_width * fib_weight * atr  # 根據波動率計算TP
        sl = max(torch.sigmoid(sl_out).item() * bb_width * fib_weight * atr, 0.002)  # 防止SL過小

        # ✅ 槓桿預測並限制在1到10倍
        leverage = min(max(torch.sigmoid(lev_out).item() * 9 + 1, 1), 10)

        # ✅ 優先使用真實reward，若無則使用模擬reward
        reward_val, _, _ = get_real_reward()
        if reward_val is None:
            reward_val = simulate_reward(direction, tp, sl, leverage, fib_distance)

        total_reward += reward_val

        # 存入回放緩衝區
        replay_buffer.add(
            state=x.squeeze(0).numpy(),
            action=action.item(),
            reward=reward_val,
            next_state=x.squeeze(0).numpy(),
            done=False
        )

        # 計算優勢
        _, next_value, _, _, _ = model(x)
        advantage = torch.tensor([reward_val], dtype=torch.float32) + GAMMA * next_value.detach() - value

        # 計算actor和critic的損失
        actor_loss = -dist.log_prob(action) * advantage.detach()
        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss

        # 更新模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ✅ 增強回放緩衝區訓練
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

    # 儲存模型和回放緩衝區
    torch.save(model.state_dict(), MODEL_PATH)
    replay_buffer.save(REPLAY_PATH)

    with torch.no_grad():
        logits, _, tp_out, sl_out, lev_out = model(x)
        probs = F.softmax(logits, dim=-1)
        confidence, selected = torch.max(probs, dim=-1)

        fib_weight = max(1 - abs(fib_distance - 0.618), 0.2)
        tp = torch.sigmoid(tp_out).item() * bb_width * fib_weight * atr
        sl = max(torch.sigmoid(sl_out).item() * bb_width * fib_weight * atr, 0.002)
        leverage = min(max(torch.sigmoid(lev_out).item() * 9 + 1, 1), 10)

    return {
        "model": "A2C",
        "direction": "Long" if selected.item() == 0 else "Short",
        "confidence": round(confidence.item(), 4),
        "tp": round(tp * 100, 2),  # 顯示為百分比
        "sl": round(sl * 100, 2),
        "leverage": int(leverage),
        "score": round(total_reward / TRAIN_STEPS, 4),
        "fib_distance": round(fib_distance, 4)
    }
