import json
import os
from datetime import datetime

LOGBOOK_PATH = "logbook.json"

def log_metrics(result, model_name="SmartTraderV16", symbol="BTC/USDT", timeframe="15m", mode="live"):
    """
    將訓練結果記錄進 logbook.json
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "mode": mode,
        "capital": result.get("capital", 0),
        "win_rate": result.get("win_rate", 0),
        "confidence": result.get("confidence", 0)
    }

    # 讀取舊資料（如果存在）
    logs = []
    if os.path.exists(LOGBOOK_PATH):
        with open(LOGBOOK_PATH, "r") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []

    # 加入新紀錄
    logs.append(entry)

    # 寫入 logbook
    with open(LOGBOOK_PATH, "w") as f:
        json.dump(logs, f, indent=2)
