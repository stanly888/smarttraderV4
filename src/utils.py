import os
import json
import numpy as np
from binance.client import Client

def fetch_real_data(config, symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1HOUR, limit=500):
    client = Client(config["binance_api_key"], config["binance_api_secret"])
    klines = client.get_historical_klines(symbol, interval, limit=limit)
    close = np.array([float(k[4]) for k in klines])
    high = np.array([float(k[2]) for k in klines])
    low = np.array([float(k[3]) for k in klines])
    volume = np.array([float(k[5]) for k in klines])
    return close, high, low, volume

def fetch_real_data_with_cache(config, symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1HOUR, limit=500, cache_file="data_cache.json"):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                raw = json.load(f)
                return (np.array(raw["close"]), np.array(raw["high"]), np.array(raw["low"]), np.array(raw["volume"]))
        except Exception:
            pass
    close, high, low, volume = fetch_real_data(config, symbol, interval, limit)
    with open(cache_file, "w") as f:
        json.dump({
            "close": close.tolist(),
            "high": high.tolist(),
            "low": low.tolist(),
            "volume": volume.tolist()
        }, f)
    return close, high, low, volume
