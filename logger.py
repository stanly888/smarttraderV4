# logger.py
import json
from datetime import datetime
import os

STATUS_FILE = "retrain_status.json"

def record_retrain_status(model_name: str, reward: float, confidence: float, source: str = "實盤"):
    """
    記錄每次 retrain 結果（包含時間、模型、信心、得分與資料來源）
    """
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model_name,
        "score": round(float(reward), 4),
        "confidence": round(float(confidence), 4),
        "source": source
    }

    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ 已更新 retrain 狀態：{model_name}（score={reward}, confidence={confidence}）")
    except Exception as e:
        print(f"❌ 無法寫入 retrain_status.json：{e}")