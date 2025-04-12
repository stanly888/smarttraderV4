from trainer import train_model
from model_selector import select_best_model
from telegram import send_strategy_update, send_daily_report
from logger import record_result, analyze_daily_log
import time

if __name__ == "__main__":
    while True:
        result = train_model()
        record_result(result)
        send_strategy_update(result)
        t = time.localtime()
        if t.tm_hour == 0 and t.tm_min < 15:
            metrics = analyze_daily_log()
            send_daily_report(metrics)
        time.sleep(900)