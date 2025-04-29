from fetch_market_data import fetch_market_data

def get_current_price(symbol="BTC-USDT") -> float:
    """
    獲取指定交易對的最新價格。
    如果無法獲取價格，則會返回 None 並輸出錯誤訊息。
    """
    try:
        # 嘗試抓取最新的市場數據
        df = fetch_market_data(symbol=symbol, interval="15m", limit=1)
        if df is not None and not df.empty:
            # 返回最新的收盤價
            return float(df["close"].iloc[-1])
        else:
            print(f"⚠️ 未能取得 {symbol} 的數據，返回 None")
            return None
    except Exception as e:
        # 捕獲錯誤並輸出錯誤訊息
        print(f"❌ 無法取得現價 ({symbol}): {e}")
        return None
