import requests
import pandas as pd

def fetch_market_data(symbol: str = "BTC-USDT", interval: str = "15m", limit: int = 100) -> pd.DataFrame:
    # Binance 期貨 API 使用 USDT 合約，symbol 需為 "BTCUSDT" 格式（無中間 dash）
    binance_symbol = symbol.replace("-", "")

    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": binance_symbol,
        "interval": interval,
        "limit": limit
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df.sort_values("timestamp").reset_index(drop=True)

    except Exception as e:
        print(f"❌ 取得 Binance K 線資料失敗: {e}")
        raise e
