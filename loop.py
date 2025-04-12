from trainer import train_model
from model_selector import select_best_model
from telegram import send_strategy_update, send_daily_report
from logger import record_result, analyze_daily_log
import time

report_sent = False  # 每日報表推播旗標

if __name__ == "__main__":
    while True:
        print("🔁 執行 train_model()...")
        result = train_model()

        # 檢查 retrain 是否正常
        if not result or "model" not in result or "confidence" not in result:
            print("❌ retrain 結果異常，略過此次推播")
            time.sleep(900)
            continue

        # Log retrain 結果
        print(f"✅ retrain 成功 → 模型：{result['model']}, 信心：{result['confidence']}, TP：{result['tp']}%, SL：{result['sl']}%")

        # 記錄與推播
        record_result(result)
        send_strategy_update(result)

        # 每日 00:00 推送日報表（僅一次）
        t = time.localtime()
        if t.tm_hour == 0 and not report_sent:
            metrics = analyze_daily_log()
            send_daily_report(metrics)
            report_sent = True
        elif t.tm_hour != 0:
            report_sent = False  # 新的一天重啟推播權限

        time.sleep(900)  # 每 15 分鐘訓練一次
