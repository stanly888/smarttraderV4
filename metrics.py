import json
import os
from datetime import datetime

LOGBOOK_FILE = "logbook_reward.json"

def analyze_daily_log() -> dict:
    if not os.path.exists(LOGBOOK_FILE):
        return {"message": "無法分析，找不到日誌檔案"}

    today = datetime.utcnow().strftime("%Y-%m-%d")
    with open(LOGBOOK_FILE, "r") as f:
        data = json.load(f)

    today_logs = [entry for entry in data if entry.get("timestamp", "").startswith(today)]
    if not today_logs:
        return {"message": "今天尚無資料"}

    total_trades = len(today_logs)
    wins = sum(1 for x in today_logs if x["reward"] > 0)
    tps = sum(1 for x in today_logs if x["reward_type"] == "TP")
    sls = sum(1 for x in today_logs if x["reward_type"] == "SL")
    avg_conf = sum(x["confidence"] for x in today_logs) / total_trades
    std_conf = (sum((x["confidence"] - avg_conf) ** 2 for x in today_logs) / total_trades) ** 0.5

    models = [x["model"] for x in today_logs]
    top_model = max(set(models), key=models.count)

    capital = 300 + sum(x["reward"] for x in today_logs)

    return {
        "final_capital": round(capital, 2),
        "win_rate": round(wins / total_trades, 2),
        "tp_rate": round(tps / total_trades, 2),
        "sl_rate": round(sls / total_trades, 2),
        "avg_confidence": round(avg_conf, 2),
        "std_confidence": round(std_conf, 4),
        "top_model": top_model,
        "total_trades": total_trades,
        "recommend_v9": avg_conf > 0.7 and capital > 300
    }