# price_fetcher.py
from fetch_market_data import fetch_market_data

def get_current_price(symbol="BTC-USDT") -> float:
    try:
        df = fetch_market_data(symbol=symbol, interval="15m", limit=1)
        return float(df["close"].iloc[-1])
    except Exception as e:
        print(f"❌ 無法取得現價：{e}")
        return None
