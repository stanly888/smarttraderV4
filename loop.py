from trainer import train_model
from model_selector import select_best_model
from telegram import send_strategy_update, send_daily_report
from logger import record_result, analyze_daily_log
import time

report_sent = False  # 每日報表推播旗標

if __name__ == "__main__":
    while True:
        result = train_model()
        record_result(result)
        send_strategy_update(result)

        t = time.localtime()
        if t.tm_hour == 0 and not report_sent:
            metrics = analyze_daily_log()
            send_daily_report(metrics)
            report_sent = True
        elif t.tm_hour != 0:
            report_sent = False  # 隔天凌晨再啟用

        time.sleep(900)
