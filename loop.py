import time
import json
import os
import logging
from datetime import datetime
import numpy as np
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
DAILY_LOSS_LIMIT = -30
DEFAULT_LEVERAGE = 5
MAX_LEVERAGE = 10
MIN_LEVERAGE = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.info("✅ SmartTrader loop 啟動成功")

report_sent = False
last_retrain_minute = -1
daily_pnl = 0
loss_triggered = False

def sanitize_inference(inference: dict) -> dict:
    cleaned = {}
    for k, v in inference.items():
        if isinstance(v, (np.float64, np.float32, np.float16)):
            cleaned[k] = float(v)
        elif isinstance(v, (np.int64, np.int32, np.int8)):
            cleaned[k] = int(v)
        elif isinstance(v, (np.bool_)):
            cleaned[k] = bool(v)
        else:
            cleaned[k] = v
    return cleaned

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

            direction = trade["direction"]
            tp_price = trade["tp_price"]
            sl_price = trade["sl_price"]
            entry_price = trade["entry_price"]

            if direction == "Long":
                if current_price >= tp_price:
                    trade["status"] = "hit_tp"
                    updated = True
                    logging.info(f"🎯 命中 TP：{trade}")
                elif current_price <= sl_price:
                    trade["status"] = "hit_sl"
                    updated = True
                    logging.info(f"⚠️ 命中 SL：{trade}")
                else:
                    if current_price <= entry_price * 0.9975:
                        trade["status"] = "hit_sl"
                        updated = True
                        logging.info(f"⚠️ 智能縮損觸發 SL：{trade}")
                    if current_price >= tp_price * 1.05:
                        trade["tp_price"] = round(current_price * 1.01, 2)
                        logging.info(f"🚀 TP自動拉伸：{trade}")
            else:
                if current_price <= tp_price:
                    trade["status"] = "hit_tp"
                    updated = True
                    logging.info(f"🎯 命中 TP：{trade}")
                elif current_price >= sl_price:
                    trade["status"] = "hit_sl"
                    updated = True
                    logging.info(f"⚠️ 命中 SL：{trade}")
                else:
                    if current_price >= entry_price * 1.0025:
                        trade["status"] = "hit_sl"
                        updated = True
                        logging.info(f"⚠️ 智能縮損觸發 SL：{trade}")
                    if current_price <= tp_price * 0.95:
                        trade["tp_price"] = round(current_price * 0.99, 2)
                        logging.info(f"🚀 TP自動拉伸：{trade}")

        if updated:
            with open(TRADES_FILE, "w") as f:
                json.dump(trades, f, indent=2)

    except Exception as e:
        logging.warning(f"❌ 檢查 open trades 錯誤：{e}")

def dynamic_leverage_adjustment(confidence: float) -> int:
    if confidence >= 0.95:
        return MAX_LEVERAGE
    elif confidence >= 0.85:
        return int(MAX_LEVERAGE * 0.8)
    elif confidence >= 0.75:
        return int(MAX_LEVERAGE * 0.6)
    else:
        return DEFAULT_LEVERAGE

# === 主迴圈開始 ===
daily_pnl = load_daily_pnl()

while True:
    now = time.localtime()
    check_open_trades()

    if now.tm_min % 15 == 0 and now.tm_min != last_retrain_minute:
        last_retrain_minute = now.tm_min
        result = train_model()

        if result.get("status") == "success":
            result["price"] = get_current_price()
            record_retrain_status(result['model'], result['score'], result['confidence'])
            log_reward_result(result)

            fib_str = f" | Fib={round(result['fib_distance'], 3)}" if "fib_distance" in result else ""
            logging.info(
                f"✅ 已完成 retrain：{result['model']} | 信心={result['confidence']:.2f} "
                f"| TP={result['tp']:.2f}% SL={result['sl']:.2f}%{fib_str}"
            )

    if not loss_triggered:
        try:
            features, (atr, bb_width, fib_distance, volatility_factor) = compute_dual_features()
            inference = train_model(features=features, atr=atr, bb_width=bb_width, fib_distance=fib_distance)

            inference = sanitize_inference(inference)

            # ✅ 加上推論狀態確認
            if inference.get("status") == "success" and inference.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
                logging.info(f"🚀 信心足夠，準備下單 | {inference}")

                adaptive_tp = inference['tp'] * volatility_factor
                adaptive_sl = inference['sl'] / volatility_factor

                if adaptive_tp < 0.002 or adaptive_tp > 0.2:
                    logging.warning(f"⚠️ TP異常({adaptive_tp:.4f}), 自動調整為0.01")
                    adaptive_tp = 0.01
                if adaptive_sl < 0.002 or adaptive_sl > 0.2:
                    logging.warning(f"⚠️ SL異常({adaptive_sl:.4f}), 自動調整為0.01")
                    adaptive_sl = 0.01

                dynamic_leverage = dynamic_leverage_adjustment(inference['confidence'])

                submit_order(
                    direction=inference['direction'],
                    tp_pct=adaptive_tp,
                    sl_pct=adaptive_sl,
                    leverage=dynamic_leverage,
                    confidence=inference['confidence']
                )

                daily_pnl += inference['score']
                save_daily_pnl(daily_pnl)

                tp_display = round(adaptive_tp * 100, 2)
                sl_display = round(adaptive_sl * 100, 2)

                message = (
                    f"📡 [SmartTrader 策略推播]\n"
                    f"模型：{inference['model']}（更新於：{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}）\n"
                    f"方向：{inference['direction']}（信心：{inference['confidence']:.2f}）\n"
                    f"槓桿：{dynamic_leverage}x\n"
                    f"TP：+{tp_display}% / SL：-{sl_display}%"
                )

                send_strategy_update({"text": message})

        except Exception as e:
            logging.warning(f"❌ 即時推論/送單失敗：{e}")

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