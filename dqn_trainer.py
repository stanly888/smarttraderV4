# dqn_trainer.py（修改 train_dqn 部分）
def train_dqn(features: np.ndarray) -> dict:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    total_reward = 0

    for _ in range(TRAIN_STEPS):
        direction_logits, tp_out, sl_out, lev_out = model(x)
        probs = torch.softmax(direction_logits, dim=-1)
        action = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, action].item()

        # 根據模型輸出決定 TP/SL/Leverage
        tp = torch.sigmoid(tp_out).item() * 3.5
        sl = torch.sigmoid(sl_out).item() * 2.0
        leverage = torch.sigmoid(lev_out).item() * 9 + 1

        reward_val, _, _ = get_real_reward()
        if reward_val is None:
            # 模擬 reward
            if action == 2:
                reward_val = np.random.uniform(-0.1, 0.2)
            else:
                hit = np.random.rand()
                raw = tp if hit < 0.5 else -sl
                fee = 0.0004 * leverage * 2
                funding = 0.00025 * leverage
                reward_val = raw * leverage - fee - funding

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
            for i in range(16):
                q_target[i, actions[i]] = rewards[i]

            loss = F.mse_loss(q_vals, q_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    save_model(model)
    buffer.save(BUFFER_PATH)

    direction = "Long" if action == 0 else "Short" if action == 1 else "Skip"

    return {
        "model": "DQN",
        "direction": direction,
        "confidence": round(confidence, 3),
        "tp": round(tp, 2),
        "sl": round(sl, 2),
        "leverage": int(leverage),
        "score": round(total_reward / TRAIN_STEPS, 4)
    }