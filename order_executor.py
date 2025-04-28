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

    # ✅ 防呆：價格無法取得時略過
    if price is None:
        print("❌ 無法獲取目前價格，略過送單")
        return

    # ✅ 防呆：價格異常過小或負值，直接略過
    if price <= 0.01:
        print(f"⚠️ 異常價格偵測：{price}，略過送單")
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
        "rewarded": False  # ✅ 預設未領取 reward
    }

    trades = []
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, "r") as f:
                trades = json.load(f)
        except Exception as e:
            print(f"⚠️ 載入交易紀錄失敗，將重新建立：{e}")
            trades = []

    trades.append(trade)

    try:
        with open(TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=2)
        print(f"✅ 模擬送單成功：方向={direction} | 進場價={price:.2f} | TP={tp_price:.2f} | SL={sl_price:.2f} | 槓桿={int(leverage)}x | 信心={confidence:.2f}")
    except Exception as e:
        print(f"❌ 儲存模擬訂單失敗：{e}")
