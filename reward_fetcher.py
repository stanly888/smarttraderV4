# reward_fetcher.py
import json
import os

TRADES_PATH = "real_trades.json"

def fetch_unrewarded_trade() -> dict:
    if not os.path.exists(TRADES_PATH):
        return None

    with open(TRADES_PATH, "r") as f:
        trades = json.load(f)

    for trade in trades:
        if trade.get("status") == "closed" and not trade.get("rewarded", False):
            return trade
    return None

def mark_trade_as_rewarded(trade_id: str):
    with open(TRADES_PATH, "r") as f:
        trades = json.load(f)

    for trade in trades:
        if trade.get("id") == trade_id:
            trade["rewarded"] = True
            break

    with open(TRADES_PATH, "w") as f:
        json.dump(trades, f, indent=2)

def compute_real_reward(trade: dict) -> tuple:
    direction = trade["direction"]
    entry = trade["entry_price"]
    exit_price = trade["exit_price"]
    leverage = float(trade.get("leverage", 1))
    
    fee = 0.0004 * leverage * 2
    funding = 0.00025 * leverage

    if direction == "Long":
        pnl = (exit_price - entry) / entry
    else:
        pnl = (entry - exit_price) / entry

    raw_reward = pnl * leverage - fee - funding

    hit_tp = exit_price >= trade["tp_price"] if direction == "Long" else exit_price <= trade["tp_price"]
    hit_sl = exit_price <= trade["sl_price"] if direction == "Long" else exit_price >= trade["sl_price"]

    return round(raw_reward, 4), hit_tp, hit_sl

def get_real_reward():
    trade = fetch_unrewarded_trade()
    if trade is None:
        return None, False, False

    reward, hit_tp, hit_sl = compute_real_reward(trade)
    mark_trade_as_rewarded(trade["id"])
    return reward, hit_tp, hit_sl
