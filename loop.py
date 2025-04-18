import time
from trainer import train_model
from telegram import send_strategy_update, send_daily_report
from logger import record_retrain_status
from metrics import analyze_daily_log
from logbook_reward import log_reward_result  # ✅ 加入 reward 模組

report_sent = False

while True:
    result = train_model()

    # ✅ 記錄訓練狀態
    record_retrain_status(result['model'], result['score'], result['confidence'])

    # ✅ 新增：記錄 TP/SL reward 到 logbook_rewards.json
    log_reward_result(result)

    # ✅ 推播策略
    send_strategy_update(result)

    # ✅ 每天凌晨推送日報
    t = time.localtime()
    if t.tm_hour == 0 and not report_sent:
        metrics = analyze_daily_log()
        send_daily_report(metrics)
        report_sent = True
    elif t.tm_hour != 0:
        report_sent = False

    # ✅ 每 15 分鐘 retrain 一次
    time.sleep(900)
