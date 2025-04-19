import pandas as pd
from datetime import datetime
from features_engineer import compute_features
from fetch_market_data import fetch_market_data
from ppo_trainer import train_ppo
from a2c_trainer import train_a2c
from dqn_trainer import train_dqn
from logger import record_retrain_status

def train_model():
    source = "實盤資料"

    try:
        df = fetch_market_data("BTC-USDT", "15m", 100)
    except Exception as e:
        print(f"❌ 無法取得實盤資料：{e}")
        return {"status": "error", "message": str(e)}

    features = compute_features(df)

    # ✅ 檢查 features 是否為空或無效
    if features is None or not features.any():
        print("⚠️ 技術指標異常，略過此次訓練")
        return {"status": "error", "message": "技術指標異常，無有效數據"}

    result_ppo = train_ppo(features)
    result_a2c = train_a2c(features)
    result_dqn = train_dqn(features)

    best = max([result_ppo, result_a2c, result_dqn], key=lambda x: x["score"])

    # ✅ 防呆機制：避免推播錯誤
    if "model" not in best or "confidence" not in best or "score" not in best:
        return {
            "status": "error",
            "message": "模型訓練結果不完整",
            "raw": best
        }

    record_retrain_status(best["model"], best["score"], best["confidence"])

    best["timestamp"] = datetime.utcnow().isoformat()
    best["status"] = "success"
    best["source"] = source
    return best
