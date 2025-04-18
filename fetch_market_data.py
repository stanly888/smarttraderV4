import requests
import pandas as pd

def fetch_market_data(symbol: str = "BTCUSDT", interval: str = "15", limit: int = 100) -> pd.DataFrame:
    url = "https://api.bybitglobal.com/v5/market/kline"
    params = {
        "category": "linear",  # 支援永續合約
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("retCode") != 0 or "result" not in data or "list" not in data["result"]:
            raise ValueError(f"Bybit 回傳異常：{data}")

        raw = data["result"]["list"]
        df = pd.DataFrame(raw, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df.sort_values("timestamp").reset_index(drop=True)

    except Exception as e:
        print(f"❌ 取得 Bybit K 線資料失敗: {e}")
        raise e
