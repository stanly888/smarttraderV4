import random
import json
import os
from typing import Union

# 設定交易資料路徑
TRADES_PATH = "real_trades.json"

def fetch_unrewarded_trade() -> Union[dict, None]:
    """取得尚未領取 reward 的已結束訂單"""
    if not os.path.exists(TRADES_PATH):
        print("⚠️ 找不到交易檔案")
        return None

    try:
        with open(TRADES_PATH, "r") as f:
            trades = json.load(f)
        
        for trade in trades:
            # 若交易已結束並且尚未領取 reward，則返回該交易
            if trade.get("status") == "closed" and not trade.get("rewarded", False):
                return trade
        return None
    except Exception as e:
        print(f"❌ 讀取交易檔案時出錯：{e}")
        return None

def mark_trade_as_rewarded(trade_id: str):
    """將指定 id 的交易標記為已領取 reward"""
    if not os.path.exists(TRADES_PATH):
        print("⚠️ 交易檔案不存在")
        return

    try:
        with open(TRADES_PATH, "r") as f:
            trades = json.load(f)

        for trade in trades:
            if trade.get("id") == trade_id:
                trade["rewarded"] = True
                break

        with open(TRADES_PATH, "w") as f:
            json.dump(trades, f, indent=2)
        print(f"✅ 交易 {trade_id} 已標記為已領取 reward")
    except Exception as e:
        print(f"❌ 標記 reward 時出錯：{e}")

def compute_real_reward(trade: dict) -> tuple[float, bool, bool]:
    """計算真實 reward 值與是否命中 TP / SL"""
    try:
        direction = trade["direction"]
        entry = trade["entry_price"]
        exit_price = trade["exit_price"]
        leverage = float(trade.get("leverage", 1))  # 使用 float 轉換槓桿倍數
        
        # 交易費用與資金費
        fee = 0.0004 * leverage * 2  # 假設手續費是0.04%
        funding = 0.00025 * leverage  # 假設資金費用是0.025%

        # 計算 PnL（盈虧）
        if direction == "Long":
            pnl = (exit_price - entry) / entry
            hit_tp = exit_price >= trade["tp_price"]
            hit_sl = exit_price <= trade["sl_price"]
        else:
            pnl = (entry - exit_price) / entry
            hit_tp = exit_price <= trade["tp_price"]
            hit_sl = exit_price >= trade["sl_price"]

        # 計算最終 reward，減去手續費和資金費
        raw_reward = pnl * leverage - fee - funding
        return round(raw_reward, 4), hit_tp, hit_sl
    except Exception as e:
        print(f"❌ 計算 reward 時出錯：{e}")
        return 0.0, False, False

def get_real_reward() -> tuple[Union[float, None], bool, bool]:
    """獲取尚未處理的 reward（如果有）"""
    trade = fetch_unrewarded_trade()
    if trade is None:
        return None, False, False  # 若無未處理交易，則返回 None

    reward, hit_tp, hit_sl = compute_real_reward(trade)  # 計算真實回報
    mark_trade_as_rewarded(trade["id"])  # 標記該交易為已領取 reward
    return reward, hit_tp, hit_sl