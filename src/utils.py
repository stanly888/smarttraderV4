
import numpy as np
import random
from binance.client import Client

def fetch_real_data(config, symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1HOUR, limit=500):
    try:
        client = Client(config["binance_api_key"], config["binance_api_secret"])
        klines = client.get_historical_klines(symbol, interval, limit=limit)
        close = np.array([float(k[4]) for k in klines])
        high = np.array([float(k[2]) for k in klines])
        low = np.array([float(k[3]) for k in klines])
        volume = np.array([float(k[5]) for k in klines])
        return close, high, low, volume
    except Exception as e:
        print("fetch_real_data error:", e)
        # fallback mock data
        close = np.array([10000 + np.sin(i/5)*100 + random.uniform(-50,50) for i in range(limit)])
        return close, close*1.01, close*0.99, np.ones_like(close) * 1000
