import requests
import pandas as pd

def fetch_market_data(symbol="BTCUSDT", interval="15", limit=100):
    """
    從 Bybit Public API 抓取 K 線資料，避免 Binance 地區封鎖。
    symbol：例如 BTCUSDT、ETHUSDT（Bybit 標準格式）
    interval：時間週期（單位分鐘）：1 / 3 / 5 / 15 / 30 / 60 / 120 / 240 / 360 / 720 / D / W / M
    """
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    if data["retCode"] != 0:
        raise Exception(f"Bybit API error: {data['retMsg']}")

    klines = data["result"]["list"]
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]
