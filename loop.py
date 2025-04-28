import time
import json
import os
import logging
from datetime import datetime
from trainer import train_model
from price_fetcher import get_current_price
from telegram import send_strategy_update, send_daily_report
from logger import record_retrain_status
from metrics import analyze_daily_log
from logbook_reward import log_reward_result
from order_executor import submit_order
from compute_dual_features import compute_dual_features

TRADES_FILE = "real_trades.json"
PNL_FILE = "daily_pnl.json"
CONFIDENCE_THRESHOLD = 0.7
DAILY_LOSS_LIMIT = -30  # ❗可以改，最大每日虧損金額限制

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.info("✅ SmartTrader loop 啟動成功")

report_sent = False
last_retrain_minute = -1
daily_pnl = 0
loss_triggered = False

def load_daily_pnl():
    if os.path.exists(PNL_FILE):
        with open(PNL_FILE, "r") as f:
            return json.load(f).get("pnl", 0)
    return 0

def save_daily_pnl(pnl):
    with open(PNL_FILE, "w") as f:
        json.dump({"pnl": pnl}, f)

def check_open_trades():
    try:
        if not os.path.exists(TRADES_FILE):
            return

        with open(TRADES_FILE, "r") as f:
            trades = json.load(f)

        updated = False
        current_price = get_current_price()
        if current_price is None:
            return

        for trade in trades:
            if trade.get("status") != "open":
                continue

            if trade["direction"] == "Long":
                if current_price >= trade["tp_price"]:
                    trade["status"] = "hit_tp"
                    updated = True
                    logging.info(f"🎯 命中 TP：{trade}")
                elif current_price <= trade["sl_price"]:
                    trade["status"] = "hit_sl"
                    updated = True
                    logging.info(f"⚠️ 命中 SL：{trade}")
            else:
                if current_price <= trade["tp_price"]:
                    trade["status"] = "hit_tp"
                    updated = True
                    logging.info(f"🎯 命中 TP：{trade}")
                elif current_price >= trade["sl_price"]:
                    trade["status"] = "hit_sl"
                    updated = True
                    logging.info(f"⚠️ 命中 SL：{trade}")

        if updated:
            with open(TRADES_FILE, "w") as f:
                json.dump(trades, f, indent=2)

    except Exception as e:
        logging.warning(f"❌ 檢查 open trades 錯誤：{e}")

# === 主迴圈 ===
daily_pnl = load_daily_pnl()

while True:
    now = time.localtime()
    check_open_trades()

    # ✅ 每 15 分鐘 retrain 一次
    if now.tm_min % 15 == 0 and now.tm_min != last_retrain_minute:
        last_retrain_minute = now.tm_min
        result = train_model()

        if result.get("status") == "success":
            result["price"] = get_current_price()
            record_retrain_status(result['model'], result['score'], result['confidence'])
            log_reward_result(result)
            send_strategy_update(result)

            fib_str = f" | Fib={round(result['fib_distance'], 3)}" if "fib_distance" in result else ""
            logging.info(
                f"✅ 已完成 retrain：{result['model']} | 信心={result['confidence']:.2f} "
                f"| TP={result['tp']:.2f}% SL={result['sl']:.2f}%{fib_str}"
            )

    # ✅ 每30秒即時推論送單
    if not loss_triggered:
        try:
            features, (atr, bb_width, fib_distance) = compute_dual_features()
            inference = train_model(features=features, atr=atr, bb_width=bb_width, fib_distance=fib_distance)

            if inference.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
                logging.info(f"🚀 信心足夠，準備下單 | {inference}")

                submit_order(
                    direction=inference['direction'],
                    tp_pct=inference['tp'],
                    sl_pct=inference['sl'],
                    leverage=inference['leverage'],
                    confidence=inference['confidence']
                )

                # 更新日累積損益
                daily_pnl += inference['score']
                save_daily_pnl(daily_pnl)

                if daily_pnl <= DAILY_LOSS_LIMIT:
                    loss_triggered = True
                    logging.warning(f"⚠️ 已達今日最大虧損，停止交易！")

                log_reward_result(inference)
                send_strategy_update(inference)

        except Exception as e:
            logging.warning(f"❌ 即時推論/送單失敗：{e}")

    # ✅ 每天00:00重置
    if now.tm_hour == 0 and not report_sent:
        metrics = analyze_daily_log()
        send_daily_report(metrics)
        logging.info("✅ 已推送日報")
        report_sent = True

        daily_pnl = 0
        save_daily_pnl(daily_pnl)
        loss_triggered = False
        logging.info("✅ 已重置當日損益與交易狀態")

    elif now.tm_hour != 0:
        report_sent = False

    time.sleep(30)
