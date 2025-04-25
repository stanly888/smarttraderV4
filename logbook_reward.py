import json
import os
import numpy as np
from datetime import datetime
import random

def simulate_trade_outcome(result: dict) -> tuple[float, str]:
    """
    隨機模擬 TP / SL 命中與報酬，用於無實盤資料情境
    """
    outcome = random.choices(["TP", "SL", "None"], weights=[0.45, 0.3, 0.25])[0]
    leverage = result.get("leverage", 1)
    reward = 0.0
    if outcome == "TP":
        reward = result["tp"] / 100 * leverage
    elif outcome == "SL":
        reward = -result["sl"] / 100 * leverage
    return round(float(reward), 4), outcome

def convert_numpy(obj):
    """處理 numpy 型別轉換為 JSON 可序列化格式"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def log_reward_result(result: dict):
    """
    將每次 retrain 的結果記錄至 logbook_reward.json
    支援真實 reward / 模擬 reward 並包含 Fib 距離、信心、價格等資訊
    """
    # 如果 reward 已被模型回傳（例如來自 reward_fetcher），則使用
    reward = result.get("score") or 0.0
    reward_type = "simulated"

    # 若還未整合真實 reward，則使用模擬 TP/SL 結果
    if result.get("score") is None:
        reward, reward_type = simulate_trade_outcome(result)

    entry = {
        "timestamp": result.get("timestamp", datetime.utcnow().isoformat()),
        "model": result.get("model", "Unknown"),
        "direction": result.get("direction", "N/A"),
        "confidence": result.get("confidence", 0.0),
        "tp": result.get("tp", 0.0),
        "sl": result.get("sl", 0.0),
        "leverage": result.get("leverage", 1),
        "reward": reward,
        "reward_type": reward_type,
        "fib_distance": result.get("fib_distance", None),
        "price": result.get("price", None),
    }

    log_file = "logbook_reward.json"
    logs = []

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            try:
                logs = json.load(f)
            except Exception:
                logs = []

    logs.append(entry)

    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2, default=convert_numpy)

    print(f"✅ 獎勵紀錄完成（reward={reward} | type={reward_type}）")