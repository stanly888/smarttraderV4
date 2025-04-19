import numpy as np
def train_dqn(features: np.ndarray) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    total_reward = 0

    for _ in range(TRAIN_STEPS):
        q_values = model(x)
        action = torch.argmax(q_values, dim=-1).item()
        reward = simulate_reward(action)
        total_reward += reward

        # 儲存到 Replay Buffer，保證 push 是一維
        buffer.push(x.squeeze(0).flatten().numpy(), action, reward)

        # 隨機取樣訓練（回放記憶）
        if len(buffer) > 16:
            batch = buffer.sample(16)

            try:
                # ✅ 加入 try-except 保護 + flatten 確保 shape 一致
                states = torch.tensor(np.stack([np.array(b[0]).flatten() for b in batch]), dtype=torch.float32)
            except ValueError as e:
                print(f"[警告] DQN 回放失敗，state shape 不一致：{e}")
                continue  # 跳過這輪訓練

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
