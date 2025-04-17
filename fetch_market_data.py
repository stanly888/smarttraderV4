import requests
import pandas as pd

def fetch_market_data(symbol="BTCUSDT", interval="15", limit=100):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    response = requests.get(url, params=params)
    
    # 嘗試轉成 JSON
    try:
        data = response.json()
    except Exception as e:
        print("❌ 回傳資料無法解析成 JSON：", response.text)
        raise e

    # 驗證成功回應
    if data["retCode"] != 0 or "result" not in data or "list" not in data["result"]:
        print("❌ API 回傳格式異常：", data)
        raise ValueError("Bybit API 回傳異常")

    raw = data["result"]["list"]

    # 建立 DataFrame
    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
    df[["open", "high", "low", "close", "volume", "turnover"]] = df[
        ["open", "high", "low", "close", "volume", "turnover"]
    ].astype(float)

    return df
