# order_executor.py
import json
import os
import uuid
from datetime import datetime
from price_fetcher import get_current_price

TRADES_FILE = "real_trades.json"

def submit_order(direction: str, tp_pct: float, sl_pct: float, leverage: float, confidence: float):
    """
    根據模型輸出方向、TP/SL 百分比、槓桿與信心，下單至模擬訂單資料。
    """
    price = get_current_price()
    if price is None:
        print("❌ 無法獲取目前價格，略過送單")
        return

    # 計算止盈與止損價格
    tp_price = price * (1 + tp_pct) if direction == "Long" else price * (1 - tp_pct)
    sl_price = price * (1 - sl_pct) if direction == "Long" else price * (1 + sl_pct)

    trade = {
        "id": str(uuid.uuid4()),  # ✅ 唯一交易識別碼
        "timestamp": datetime.utcnow().isoformat(),
        "direction": direction,
        "confidence": round(confidence, 4),
        "entry_price": round(price, 2),
        "tp": round(tp_pct, 4),
        "sl": round(sl_pct, 4),
        "leverage": int(leverage),
        "tp_price": round(tp_price, 2),
        "sl_price": round(sl_price, 2),
        "status": "open",
        "rewarded": False         # ✅ 預設未領取 reward
    }

    trades = []
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, "r") as f:
            trades = json.load(f)

    trades.append(trade)

    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2)

    print(f"✅ 新增模擬訂單：{trade}")