import os
import json
import numpy as np
from binance.client import Client
import random
import torch

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

    required_keys = ["binance_api_key", "binance_api_secret", "telegram_bot_token", "telegram_chat_id"]
    missing_keys = [k for k in required_keys if not config.get(k)]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    return config

def fetch_real_data(config, mode="live", symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1HOUR, limit=500):
    client = Client(config["binance_api_key"], config["binance_api_secret"])
    klines = client.get_historical_klines(symbol, interval, limit=limit)
    close = np.array([float(k[4]) for k in klines])
    high = np.array([float(k[2]) for k in klines])
    low = np.array([float(k[3]) for k in klines])
    volume = np.array([float(k[5]) for k in klines])
    return close, high, low, volume

def get_random_historical_data(close, high, low, volume, segment_length=100):
    start_idx = random.randint(0, len(close) - segment_length)
    return (
        close[start_idx:start_idx + segment_length],
        high[start_idx:start_idx + segment_length],
        low[start_idx:start_idx + segment_length],
        volume[start_idx:start_idx + segment_length]
    )

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_model(model, path):
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)

def load_model_if_exists(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Loaded model from {path}")
    return model
