import json
import os
import logging
from datetime import datetime
from src.multi_trainer import MultiStrategyTrainer
from src.utils import fetch_real_data, get_random_historical_data
from src.performance import log_strategy_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config():
    try:
        with open("config/config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Config file 'config/config.json' not found.")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in config file.")
    
    for key in ["binance_api_key", "binance_api_secret", "telegram_bot_token", "telegram_chat_id"]:
        config[key] = os.getenv(key, config.get(key))
    
    missing_keys = [k for k in ["binance_api_key", "binance_api_secret"] if not config.get(k)]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    return config

def fetch_training_data(config, hist_close, hist_high, hist_low, is_morning):
    if is_morning:
        return hist_close[-50:], hist_high[-50:], hist_low[-50:]
    return get_random_historical_data(hist_close, hist_high, hist_low)

def main():
    config = load_config()
    try:
        hist_close, hist_high, hist_low = fetch_real_data(config, limit=500)
    except Exception as e:
        logging.error(f"❌ Failed to fetch historical data: {e}")
        return

    trainer = MultiStrategyTrainer(hist_close, hist_high, hist_low, config)

    now = datetime.now()
    m_start = config.get("morning_start_hour", 0)
    m_end = config.get("morning_end_hour", 12)
    is_morning = m_start <= now.hour < m_end

    try:
        close, high, low = fetch_training_data(config, hist_close, hist_high, hist_low, is_morning)
        best_result = trainer.train_all(close, high, low, is_morning)
        log_strategy_summary(best_result, config)
        logging.info(f"✅ Strategy complete. Best: {best_result['model']} | Capital: {best_result.get('capital')}")
    except Exception as e:
        logging.error(f"❌ Error during training or summary: {e}")

if __name__ == "__main__":
    main()
