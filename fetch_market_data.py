import requests
import pandas as pd

def fetch_market_data(symbol: str = "BTC-USDT", interval: str = "15m", limit: int = 100) -> pd.DataFrame:
    url = "https://open-api.bingx.com/openApi/spot/v1/market/kline"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("code") != 0 or "data" not in data:
            raise ValueError(f"BingX 回傳異常：{data}")

        raw = data["data"]
        df = pd.DataFrame(raw, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "quoteVolume", "count"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df.sort_values("timestamp").reset_index(drop=True)

    except Exception as e:
        print(f"❌ 取得 BingX K 線資料失敗: {e}")
        raise e
