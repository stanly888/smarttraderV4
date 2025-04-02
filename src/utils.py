import os
import json
import datetime
import random

def load_config():
    try:
        with open("config/config.json") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Config file 'config/config.json' not found.")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in config file.")
    
    # 覆蓋為 Render 環境變數優先
    for k in ["binance_api_key", "binance_api_secret", "telegram_bot_token", "telegram_chat_id"]:
        config[k] = os.getenv(k, config.get(k))

    required_keys = ["binance_api_key", "binance_api_secret", "telegram_bot_token", "telegram_chat_id"]
    missing_keys = [k for k in required_keys if not config.get(k)]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    return config


def fetch_real_data(symbol="BTC/USDT", interval="15m", limit=100):
    """
    模擬從交易所取得歷史資料。實際版本請改為使用 ccxt 或 binance API。
    """
    now = datetime.datetime.now()
    data = []
    base_price = 30000
    for i in range(limit):
        candle_time = now - datetime.timedelta(minutes=15 * i)
        price = base_price + random.uniform(-500, 500)
        data.append({
            "timestamp": candle_time.strftime("%Y-%m-%d %H:%M:%S"),
            "open": price * random.uniform(0.99, 1.01),
            "high": price * random.uniform(1.00, 1.03),
            "low": price * random.uniform(0.97, 1.00),
            "close": price,
            "volume": random.uniform(5, 50)
        })
    return list(reversed(data))  # 時間由舊到新
