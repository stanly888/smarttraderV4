import json
import os
from datetime import datetime
from src.trainer import PPOTrainer
from src.utils import fetch_real_data, get_random_historical_data

def load_config():
    with open("config/config.json", "r") as f:
        config = json.load(f)
    keys = ["binance_api_key", "binance_api_secret", "telegram_bot_token", "telegram_chat_id"]
    for key in keys:
        if os.getenv(key):
            config[key] = os.getenv(key)
    return config

def main():
    config = load_config()
    historical_close, historical_high, historical_low = fetch_real_data(config, limit=500)
    trainer = PPOTrainer(historical_close, historical_high, historical_low, config)

    now = datetime.now()
    is_morning = now.hour < 12
    if is_morning:
        close, high, low = fetch_real_data(config, limit=50)
    else:
        close, high, low = get_random_historical_data(historical_close, historical_high, historical_low)
    
    trainer.train(close, high, low, is_morning)

if __name__ == "__main__":
    main()
