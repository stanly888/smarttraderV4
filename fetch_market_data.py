import ccxt
import pandas as pd

def fetch_market_data(symbol, timeframe, limit):
    exchange = ccxt.binance({
        'enableRateLimit': True  # 不要加 apiKey/secret，避免觸發限制
    })
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
