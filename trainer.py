# trainer.py
import pandas as pd
import random
from datetime import datetime
from features_engineer import compute_features
from fetch_market_data import fetch_market_data
from fetch_random_historical_data import fetch_random_historical_data
from ppo_trainer import train_ppo
from a2c_trainer import train_a2c
from dqn_trainer import train_dqn
from logger import record_retrain_status

def train_model():
    # ✅ 隨機選擇資料來源：BingX 實盤 or Bybit 歷史
    use_historical = random.random() < 0.5
    source = "歷史資料" if use_historical else "實盤資料"

    try:
        if use_historical:
            df = fetch_random_historical_data("BTCUSDT", "15", 100)  # Bybit 用 BTCUSDT + 數字分鐘
        else:
            df = fetch_market_data("BTC-USDT", "15m", 100)  # BingX 用 BTC-USDT + 15m
    except Exception as e:
        print(f"❌ 無法取得 {source}：{e}")
        return {"status": "error", "message": str(e)}

    features = compute_features(df)

    result_ppo = train_ppo(features)
    result_a2c = train_a2c(features)
    result_dqn = train_dqn(features)

    best = max([result_ppo, result_a2c, result_dqn], key=lambda x: x["score"])
    record_retrain_status(best["model"], best["score"], best["confidence"])

    best["timestamp"] = datetime.utcnow().isoformat()
    best["status"] = "success"
    best["source"] = source  # 記錄來源
    return best
