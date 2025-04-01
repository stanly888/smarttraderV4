import json, os, logging
from datetime import datetime
from src.multi_trainer import MultiStrategyTrainer
from src.utils import fetch_real_data_with_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    try:
        with open("config/config.json") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Config file 'config/config.json' not found.")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in config file.")
    
    for k in ["binance_api_key", "binance_api_secret", "telegram_bot_token", "telegram_chat_id"]:
        config[k] = os.getenv(k, config.get(k))
    
    required_keys = ["binance_api_key", "binance_api_secret"]
    missing = [k for k in required_keys if not config.get(k)]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    return config

def main():
    try:
        config = load_config()
        mode = os.getenv("MODE", "train").lower()
        valid_modes = {"train", "backtest"}
        if mode not in valid_modes:
            logging.warning(f"Invalid MODE: {mode}. Defaulting to 'train'.")
            mode = "train"

        data_limit = config.get("data_limit", {"train": 500, "backtest": 1000}).get(mode, 500)
        logging.info(f"Starting in {mode} mode | Data limit: {data_limit}")
        close, high, low, volume = fetch_real_data_with_cache(config, limit=data_limit)
        logging.info(f"Fetched data: {len(close)} entries")

        trainer = MultiStrategyTrainer(config)
        if mode == "backtest":
            result = trainer.backtest(close, high, low, volume)
            logging.info(f"Backtest completed: {result}")
        else:
            result = trainer.train_all(close, high, low, volume)
            logging.info(f"Training completed: {result}")
    except Exception as e:
        logging.error(f"Execution failed: {e}")

if __name__ == "__main__":
    main()
