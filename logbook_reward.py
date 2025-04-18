# logbook_reward.py
import json
from datetime import datetime
import os

def simulate_trade_outcome(result: dict) -> float:
    """
    根據模型輸出模擬是否命中 TP / SL 並產生 reward。
    """
    # 隨機模擬是否成功命中 TP 或 SL
    import random
    outcome = random.choices(["tp", "sl", "none"], weights=[0.45, 0.3, 0.25])[0]  # 可調整機率
    
    leverage = result.get("leverage", 1)
    reward = 0.0

    if outcome == "tp":
        reward = result["tp"] / 100 * leverage
    elif outcome == "sl":
        reward = -result["sl"] / 100 * leverage

    return round(reward, 4)

def log_reward_result(result: dict):
    """
    將 reward 紀錄進 logbook_rewards.json
    """
    reward = simulate_trade_outcome(result)
    entry = {
        "timestamp": result.get("timestamp", datetime.utcnow().isoformat()),
        "model": result["model"],
        "direction": result["direction"],
        "confidence": result["confidence"],
        "tp": result["tp"],
        "sl": result["sl"],
        "leverage": result["leverage"],
        "reward": reward
    }

    log_file = "logbook_rewards.json"
    logs = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)

    logs.append(entry)

    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"✅ 獎勵紀錄完成（reward={reward}）")
