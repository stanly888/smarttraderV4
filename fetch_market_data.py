import requests
import pandas as pd

def fetch_market_data(symbol: str = "BTCUSDT", interval: str = "15", limit: int = 100) -> pd.DataFrame:
    # ✅ 使用替代 endpoint 避開 CloudFront 封鎖
    url = f"https://api2.bybit.com/market/kline?symbol={symbol}&interval={interval}&limit={limit}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        raw = data.get("result", [])
        if not raw or len(raw[0]) < 6:
            raise ValueError("Bybit 回傳資料格式異常或為空")

        df = pd.DataFrame(raw, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df.sort_values("timestamp").reset_index(drop=True)

    except Exception as e:
        print(f"❌ 取得 Bybit K 線資料失敗: {e}")
        raise e
