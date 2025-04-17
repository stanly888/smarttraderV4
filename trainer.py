import pandas as pd
from datetime import datetime
from features_engineer import compute_features
from fetch_market_data import fetch_market_data
from ppo_trainer import train_ppo
from a2c_trainer import train_a2c
from dqn_trainer import train_dqn
from logger import record_retrain_status

def train_model():
    df = fetch_market_data("BTC/USDT", "15m", 100)
    features = compute_features(df)
    result_ppo = train_ppo(features)
    result_a2c = train_a2c(features)
    result_dqn = train_dqn(features)
    best = max([result_ppo, result_a2c, result_dqn], key=lambda x: x["score"])
    record_retrain_status(best["model"], best["score"], best["confidence"])
    best["timestamp"] = datetime.utcnow().isoformat()
    best["status"] = "success"
    return best