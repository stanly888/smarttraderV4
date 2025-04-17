import time
from trainer import train_model
from telegram import send_strategy_update, send_daily_report
from logger import record_retrain_status
from metrics import analyze_daily_log

report_sent = False
while True:
    result = train_model()
    record_retrain_status(result['model'], result['score'], result['confidence'])
    send_strategy_update(result)
    t = time.localtime()
    if t.tm_hour == 0 and not report_sent:
        metrics = analyze_daily_log()
        send_daily_report(metrics)
        report_sent = True
    elif t.tm_hour != 0:
        report_sent = False
    time.sleep(900)