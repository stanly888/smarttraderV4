import time
import json
import os
import logging
from datetime import datetime
from trainer import train_model
from price_fetcher import get_current_price        # ✅ 從 price_fetcher 分離引用
from order_executor import submit_order            # ✅ 保留送單模組
from telegram import send_strategy_update, send_daily_report
from logger import record_retrain_status
from metrics import analyze_daily_log
from logbook_reward import log_reward_result

TRADES_FILE = "real_trades.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.info("✅ SmartTrader loop 啟動成功")

report_sent = False
last_retrain_minute = -1

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
while True:
    now = time.localtime()

    # ✅ 每分鐘檢查 TP/SL 命中
    check_open_trades()

    # ✅ 每 15 分 retrain 並推播
    if now.tm_min % 15 == 0 and now.tm_min != last_retrain_minute:
        last_retrain_minute = now.tm_min
        result = train_model()

        if result.get("status") == "success":
            record_retrain_status(result['model'], result['score'], result['confidence'])
            log_reward_result(result)
            send_strategy_update(result)

            logging.info(f"✅ 已完成訓練與推播：{result['model']} | 信心={result['confidence']:.2f} | TP={result['tp']}% SL={result['sl']}%")

            # ✅ 執行模擬送單
            submit_order(
                direction=result['direction'],
                tp_pct=result['tp'],
                sl_pct=result['sl'],
                leverage=result['leverage'],
                confidence=result['confidence']
            )

        else:
            logging.warning(f"❌ 本輪訓練失敗：{result.get('message')}")

    # ✅ 每日 00:00 推播日報（僅一次）
    if now.tm_hour == 0 and not report_sent:
        metrics = analyze_daily_log()
        send_daily_report(metrics)
        logging.info("✅ 已推送日報")
        report_sent = True
    elif now.tm_hour != 0:
        report_sent = False

    time.sleep(60)
