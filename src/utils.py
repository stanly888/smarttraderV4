import os
import json

def load_config():
    try:
        with open("config/config.json") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Config file 'config/config.json' not found.")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in config file.")
    
    # 環境變數覆蓋（Render）
    for k in ["binance_api_key", "binance_api_secret", "telegram_bot_token", "telegram_chat_id"]:
        config[k] = os.getenv(k, config.get(k))

    # 必要欄位驗證
    required_keys = ["binance_api_key", "binance_api_secret", "telegram_bot_token", "telegram_chat_id"]
    missing_keys = [k for k in required_keys if not config.get(k)]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    return config
