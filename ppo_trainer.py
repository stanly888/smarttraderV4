import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from ppo_model import UnifiedRLModel, save_model, load_model_if_exists
from replay_buffer import ReplayBuffer
from reward_fetcher import get_real_reward
from reward_utils import simulate_reward  # ✅ 引入外部 simulate_reward 函數

TRAIN_STEPS = 20
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 8

MODEL_PATH = "ppo_model.pt"
REPLAY_PATH = "ppo_replay.json"

# 初始化模型與優化器
model = UnifiedRLModel(input_dim=35)
load_model_if_exists(model, MODEL_PATH)
optimizer = optim.Adam(model.parameters(), lr=LR)

# 初始化回放緩衝區
replay_buffer = ReplayBuffer(capacity=1000)
replay_buffer.load(REPLAY_PATH)
if len(replay_buffer) > 0:
    print("✅ PPO 回放緩衝區已載入")

def train_ppo(features: np.ndarray, atr: float, bb_width: float, fib_distance: float, confidence_threshold: float = 0.7) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # 將特徵轉換為 tensor
    total_reward = 0

    for _ in range(TRAIN_STEPS):
        # 模型前向傳播，得到模型的輸出
        logits, value, tp_out, sl_out, lev_out = model(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()  # 從分佈中抽取一個動作

        direction = "Long" if action.item() == 0 else "Short"
        confidence = probs[0, action].item()

        # ✅ 信心過濾：如果信心低於設定的閾值，就選擇 Skip
        if confidence < confidence_threshold:
            action = torch.tensor(1)  # 設定為 Skip
            confidence = 1.0  # 避免影響後續計算出錯

        # ✅ TP/SL 動態調整
        fib_weight = max(1 - abs(fib_distance - 0.618), 0.2)  # 根據斐波那契進行加權
        
        # 確保 tp_out, sl_out, lev_out 都是 tensor 類型再使用 sigmoid
        tp_out = torch.tensor(tp_out, dtype=torch.float32)  # 確保是 tensor 類型
        sl_out = torch.tensor(sl_out, dtype=torch.float32)  # 確保是 tensor 類型
        lev_out = torch.tensor(lev_out, dtype=torch.float32)  # 確保是 tensor 類型

        tp = torch.sigmoid(tp_out).item() * bb_width * fib_weight * atr  # 根據波動率計算TP，轉為 float
        sl = max(torch.sigmoid(sl_out).item() * bb_width * fib_weight * atr, 0.002)  # 防止SL過小，轉為 float

        # ✅ 槓桿動態調整，限制在 1 到 10 倍之間
        leverage = min(max(torch.sigmoid(lev_out).item() * 9 + 1, 1), 10)

        # ✅ 優先使用真實的回報
        reward_val, _, _ = get_real_reward()
        if reward_val is None:
            reward_val = simulate_reward(
                direction=direction,
                tp=tp,
                sl=sl,
                leverage=leverage,
                fib_distance=fib_distance
            )

        total_reward += reward_val

        # 存儲回放緩衝區
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

        # 計算損失
        actor_loss = -dist.log_prob(action) * advantage.detach()
        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss

        # 更新模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 增強回放緩衝區訓練
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
                print(f"⚠️ PPO 回放訓練失敗：{e}")

    # 儲存模型和回放緩衝區
    torch.save(model.state_dict(), MODEL_PATH)
    replay_buffer.save(REPLAY_PATH)

    with torch.no_grad():
        logits, _, tp_out, sl_out, lev_out = model(x)
        probs = F.softmax(logits, dim=-1)
        confidence, selected = torch.max(probs, dim=-1)

        fib_weight = max(1 - abs(fib_distance - 0.618), 0.2)
        tp = float(torch.sigmoid(tp_out).item()) * bb_width * fib_weight * atr  # 轉為 float
        sl = max(float(torch.sigmoid(sl_out).item()) * bb_width * fib_weight * atr, 0.002)  # 轉為 float
        leverage = min(max(torch.sigmoid(lev_out).item() * 9 + 1, 1), 10)

    return {
        "model": "PPO",
        "direction": "Long" if selected.item() == 0 else "Short",
        "confidence": round(confidence, 4),
        "tp": round(tp * 100, 2),  # 百分比顯示
        "sl": round(sl * 100, 2),
        "leverage": int(leverage),
        "score": round(reward_val, 4),
        "fib_distance": round(fib_distance, 4)
    }