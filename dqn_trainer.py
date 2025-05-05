import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from dqn_model import DQN, save_model, load_model_if_exists
from replay_buffer import ReplayBuffer
from reward_fetcher import get_real_reward
from reward_utils import simulate_reward  # 引入外部 simulate_reward 函數

MODEL_PATH = "dqn_model.pt"
BUFFER_PATH = "dqn_replay.json"
TRAIN_STEPS = 20
BATCH_SIZE = 16
LR = 1e-3

# 初始化模型與優化器
model = DQN(input_dim=35)
optimizer = optim.Adam(model.parameters(), lr=LR)

# 載入模型與回放緩衝區
load_model_if_exists(model, MODEL_PATH)
print("✅ DQN 模型已載入")

buffer = ReplayBuffer(capacity=1000)
buffer.load(BUFFER_PATH)
if len(buffer) > 0:
    print("✅ DQN Replay Buffer 已載入")

def train_dqn(features: np.ndarray, atr: float, bb_width: float, fib_distance: float, confidence_threshold: float = 0.7) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    total_reward = 0

    for _ in range(TRAIN_STEPS):
        # 進行前向傳播，得到模型的輸出
        direction_logits, tp_out, sl_out, lev_out = model(x)
        probs = torch.softmax(direction_logits, dim=-1)
        action = torch.argmax(probs, dim=-1).item()  # 取得最佳的動作
        confidence = probs[0, action].item()

        # ✅ 信心過濾：信心不夠高直接選擇 Skip
        if confidence < confidence_threshold:
            action = 2  # 直接設定為 Skip
            confidence = 1.0  # 避免影響後續計算

        # ✅ 動態 TP/SL 計算，根據 fib_weight 微調
        fib_weight = max(1 - abs(fib_distance - 0.618), 0.2)  # 根據斐波那契進行加權
        
        # 確保 tp_out, sl_out, lev_out 都是 tensor 類型再使用 sigmoid
        tp_out = torch.tensor(tp_out, dtype=torch.float32)  # 確保是 tensor 類型
        sl_out = torch.tensor(sl_out, dtype=torch.float32)  # 確保是 tensor 類型
        lev_out = torch.tensor(lev_out, dtype=torch.float32)  # 確保是 tensor 類型

        tp = torch.sigmoid(tp_out).item() * bb_width * fib_weight * atr  # 根據波動率計算TP，轉為 float
        sl = max(torch.sigmoid(sl_out).item() * bb_width * fib_weight * atr, 0.002)  # 防止SL過小，轉為 float

        # ✅ 槓桿預測並限制在1~10倍
        leverage = min(max(torch.sigmoid(lev_out).item() * 9 + 1, 1), 10)

        # ✅ 優先使用真實 reward，若無則使用模擬
        reward_val, _, _ = get_real_reward()
        if reward_val is None:
            reward_val = simulate_reward(
                direction="Long" if action == 0 else "Short" if action == 1 else "Skip",
                tp=tp,
                sl=sl,
                leverage=leverage,
                fib_distance=fib_distance
            )

        total_reward += reward_val

        # 存儲回放緩衝區
        buffer.add(
            state=x.squeeze(0).numpy(),
            action=action,
            reward=reward_val,
            next_state=x.squeeze(0).numpy(),
            done=False
        )

        # 回放練習
        if len(buffer) >= BATCH_SIZE:
            try:
                states, actions, rewards, _, _ = buffer.sample(BATCH_SIZE)
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)

                direction_logits, _, _, _ = model(states)
                q_vals = direction_logits
                q_target = q_vals.detach().clone()

                for i in range(len(rewards)):
                    q_target[i, actions[i]] = rewards[i]

                loss = F.mse_loss(q_vals, q_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"⚠️ DQN 回放訓練失敗：{e}")

    # 儲存模型與回放緩衝區
    save_model(model, MODEL_PATH)
    buffer.save(BUFFER_PATH)

    direction = "Long" if action == 0 else "Short" if action == 1 else "Skip"

    return {
        "model": "DQN",
        "direction": direction,
        "confidence": round(confidence, 4),
        "tp": round(tp * 100, 2),  # 統一顯示為百分比
        "sl": round(sl * 100, 2),
        "leverage": int(leverage),
        "score": round(total_reward / TRAIN_STEPS, 4),
        "fib_distance": round(fib_distance, 4)
    }