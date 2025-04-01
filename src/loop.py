
import os
import json
import time
import traceback
from datetime import datetime
from src.multi_trainer import MultiStrategyTrainer
from src.utils import fetch_real_data
from src.logger import log_strategy_summary

def load_config():
    with open("config/config.json", "r") as f:
        config = json.load(f)
    for key in ["binance_api_key", "binance_api_secret", "telegram_bot_token", "telegram_chat_id"]:
        config[key] = os.getenv(key, config.get(key))
    return config

if __name__ == "__main__":
    while True:
        try:
            config = load_config()
            now = datetime.utcnow()
            is_morning = now.hour < 12

            data = fetch_real_data(config, limit=500)
            if is_morning:
                # 用最後 50 根真實資料模擬實盤 retrain
                close, high, low, volume = [x[-50:] for x in data]
            else:
                close, high, low, volume = data

            trainer = MultiStrategyTrainer(config)
            result = trainer.train_all(close, high, low, volume, is_morning)
            log_strategy_summary(result, config)
        except Exception as e:
            print("Loop Error:", e)
            traceback.print_exc()

        time.sleep(3600)
