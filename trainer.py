# trainer.py
import pandas as pd
from datetime import datetime
from pytz import timezone
from features_engineer import compute_features
from fetch_market_data import fetch_market_data
from fetch_random_historical_data import fetch_random_historical_data
from ppo_trainer import train_ppo
from a2c_trainer import train_a2c
from dqn_trainer import train_dqn
from logger import record_retrain_status

def train_model():
    # ✅ 根據台灣時間自動切換資料來源
    local_time = datetime.now(timezone("Asia/Taipei"))
    hour = local_time.hour
    use_historical = hour < 12  # 0~11 點：歷史，12~23 點：實盤
    source = "歷史資料" if use_historical else "實盤資料"

    try:
        if use_historical:
            df = fetch_random_historical_data("BTCUSDT", "15", 100)
        else:
            df = fetch_market_data("BTC-USDT", "15m", 100)
    except Exception as e:
        print(f"❌ 無法取得 {source}：{e}")
        return {"status": "error", "message": str(e)}

    features = compute_features(df)

    # ✅ 新增：檢查 features 是否為空或無效
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
