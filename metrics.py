# metrics.py
import json
import os
from datetime import datetime

LOGBOOK_FILE = "logbook_reward.json"

def analyze_daily_log() -> dict:
    if not os.path.exists(LOGBOOK_FILE):
        return {"message": "❌ 找不到 logbook_reward.json"}

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    try:
        with open(LOGBOOK_FILE, "r") as f:
            all_logs = json.load(f)
    except Exception as e:
        return {"message": f"❌ 無法讀取 reward 日誌：{e}"}

    today_logs = [log for log in all_logs if log.get("timestamp", "").startswith(today_str)]
    if not today_logs:
        return {"message": "⚠️ 今日尚無交易紀錄"}

    total_trades = len(today_logs)
    total_reward = sum(log["reward"] for log in today_logs)
    capital = 300 + total_reward

    wins = sum(1 for log in today_logs if log["reward"] > 0)
    tp_hits = sum(1 for log in today_logs if log.get("reward_type") == "TP")
    sl_hits = sum(1 for log in today_logs if log.get("reward_type") == "SL")
    avg_conf = sum(log["confidence"] for log in today_logs) / total_trades
    std_conf = (sum((log["confidence"] - avg_conf) ** 2 for log in today_logs) / total_trades) ** 0.5

    model_counts = {}
    for log in today_logs:
        model = log.get("model", "Unknown")
        model_counts[model] = model_counts.get(model, 0) + 1
    top_model = max(model_counts, key=model_counts.get)

    return {
        "final_capital": round(capital, 2),
        "win_rate": round(wins / total_trades, 2),
        "tp_rate": round(tp_hits / total_trades, 2),
        "sl_rate": round(sl_hits / total_trades, 2),
        "avg_confidence": round(avg_conf, 2),
        "std_confidence": round(std_conf, 4),
        "top_model": top_model,
        "total_trades": total_trades,
        "recommend_v9": avg_conf > 0.7 and capital > 300
    }