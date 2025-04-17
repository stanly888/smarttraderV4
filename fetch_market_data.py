import ccxt
import pandas as pd
import os

def fetch_market_data(symbol, timeframe, limit):
    exchange = ccxt.binance({
        'apiKey': os.getenv('binance_api_key'),
        'secret': os.getenv('binance_api_secret'),
        'enableRateLimit': True
    })
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df