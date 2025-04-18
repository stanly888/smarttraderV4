# fetch_random_historical_data.py
import requests
import pandas as pd
import random
from datetime import datetime, timedelta

def fetch_random_historical_data(symbol="BTCUSDT", interval="15", limit=100):
    url = "https://api.bybit.com/v5/market/kline"
    end_time = int((datetime.utcnow() - timedelta(days=random.randint(10, 180))).timestamp() * 1000)

    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "end": end_time
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("retCode") != 0 or "result" not in data or "list" not in data["result"]:
            raise ValueError(f"歷史資料回傳異常：{data}")

        raw = data["result"]["list"]
        df = pd.DataFrame(raw, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df.sort_values("timestamp").reset_index(drop=True)

    except Exception as e:
        print(f"❌ 抓取歷史資料失敗：{e}")
        raise e
