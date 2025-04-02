import numpy as np
from binance.client import Client

def fetch_real_data(config, symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_15MINUTE, limit=200):
    client = Client(config["binance_api_key"], config["binance_api_secret"])
    klines = client.get_historical_klines(symbol, interval, limit=limit)
    close = np.array([float(k[4]) for k in klines])
    high = np.array([float(k[2]) for k in klines])
    low = np.array([float(k[3]) for k in klines])
    volume = np.array([float(k[5]) for k in klines])
    return close, high, low, volume
