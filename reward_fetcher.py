# reward_fetcher.py
import json
import os

TRADES_PATH = "real_trades.json"

def fetch_unrewarded_trade() -> dict | None:
    """取得尚未領取 reward 的已結束訂單"""
    if not os.path.exists(TRADES_PATH):
        return None

    with open(TRADES_PATH, "r") as f:
        trades = json.load(f)

    for trade in trades:
        if trade.get("status") == "closed" and not trade.get("rewarded", False):
            return trade
    return None

def mark_trade_as_rewarded(trade_id: str):
    """將指定 id 的交易標記為已領取 reward"""
    if not os.path.exists(TRADES_PATH):
        return

    with open(TRADES_PATH, "r") as f:
        trades = json.load(f)

    for trade in trades:
        if trade.get("id") == trade_id:
            trade["rewarded"] = True
            break

    with open(TRADES_PATH, "w") as f:
        json.dump(trades, f, indent=2)

def compute_real_reward(trade: dict) -> tuple[float, bool, bool]:
    """計算真實 reward 值與是否命中 TP / SL"""
    try:
        direction = trade["direction"]
        entry = trade["entry_price"]
        exit_price = trade["exit_price"]
        leverage = float(trade.get("leverage", 1))
        
        fee = 0.0004 * leverage * 2
        funding = 0.00025 * leverage

        if direction == "Long":
            pnl = (exit_price - entry) / entry
            hit_tp = exit_price >= trade["tp_price"]
            hit_sl = exit_price <= trade["sl_price"]
        else:
            pnl = (entry - exit_price) / entry
            hit_tp = exit_price <= trade["tp_price"]
            hit_sl = exit_price >= trade["sl_price"]

        raw_reward = pnl * leverage - fee - funding
        return round(raw_reward, 4), hit_tp, hit_sl
    except Exception as e:
        print(f"❌ Reward 計算錯誤：{e}")
        return 0.0, False, False

def get_real_reward() -> tuple[float | None, bool, bool]:
    """獲取尚未處理的 reward（如果有）"""
    trade = fetch_unrewarded_trade()
    if trade is None:
        return None, False, False

    reward, hit_tp, hit_sl = compute_real_reward(trade)
    mark_trade_as_rewarded(trade["id"])
    return reward, hit_tp, hit_sl