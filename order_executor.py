# order_executor.py
import json
import os
from datetime import datetime
from fetch_market_data import fetch_market_data

TRADES_FILE = "real_trades.json"

def get_current_price(symbol="BTC-USDT") -> float:
    """從市場抓取當前價格（預設為 close）"""
    try:
        df = fetch_market_data(symbol=symbol, interval="15m", limit=1)
        return float(df["close"].iloc[-1])
    except Exception as e:
        print(f"❌ 無法取得現價：{e}")
        return None

def submit_order(direction: str, tp_pct: float, sl_pct: float, leverage: float, confidence: float,
                 symbol: str = "BTC-USDT") -> bool:
    """
    建立一筆模擬交易：
    - direction: "Long" 或 "Short"
    - tp_pct/sl_pct: 模型預測的 TP / SL 百分比
    - leverage: 使用槓桿
    - confidence: 模型信心
    """
    entry_price = get_current_price(symbol)
    if entry_price is None:
        return False

    # 計算 TP/SL 價
    if direction == "Long":
        tp_price = entry_price * (1 + tp_pct / 100)
        sl_price = entry_price * (1 - sl_pct / 100)
    else:
        tp_price = entry_price * (1 - tp_pct / 100)
        sl_price = entry_price * (1 + sl_pct / 100)

    order = {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "direction": direction,
        "entry": round(entry_price, 2),
        "tp_price": round(tp_price, 2),
        "sl_price": round(sl_price, 2),
        "tp_pct": round(tp_pct, 2),
        "sl_pct": round(sl_pct, 2),
        "leverage": round(leverage, 2),
        "confidence": round(confidence, 4),
        "status": "open",  # open, hit_tp, hit_sl
    }

    try:
        if os.path.exists(TRADES_FILE):
            with open(TRADES_FILE, "r") as f:
                trades = json.load(f)
        else:
            trades = []

        trades.append(order)

        with open(TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=2)

        print(f"✅ 模擬交易已送出：{order}")
        return True
    except Exception as e:
        print(f"❌ 無法寫入交易：{e}")
        return False
