import time
import logging
from trainer import train_model
from telegram import send_strategy_update, send_daily_report
from logger import record_retrain_status
from metrics import analyze_daily_log
from logbook_reward import log_reward_result

# 啟用日誌系統（可顯示於 Render 日誌面板）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.info("✅ SmartTrader loop 啟動成功")

report_sent = False

while True:
    result = train_model()

    if result.get("status") == "success":
        # 記錄 retrain 狀態
        record_retrain_status(result['model'], result['score'], result['confidence'])

        # 記錄 TP/SL reward
        log_reward_result(result)

        # 推播到 Telegram
        send_strategy_update(result)

        # 顯示 retrain 日誌
        logging.info(f"✅ 已完成訓練與推播：{result['model']} | 信心={result['confidence']:.2f} | TP={result['tp']}% SL={result['sl']}%")
    else:
        logging.warning(f"❌ 本輪訓練失敗：{result.get('message')}")

    # 每日 00:00 推播日報（僅一次）
    t = time.localtime()
    if t.tm_hour == 0 and not report_sent:
        metrics = analyze_daily_log()
        send_daily_report(metrics)
        logging.info("✅ 已推送日報")
        report_sent = True
    elif t.tm_hour != 0:
        report_sent = False

    # 每 15 分鐘 retrain 一次
    time.sleep(900)
