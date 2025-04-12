import schedule
import time
import logging
import os
from datetime import date, datetime
from typing import Dict, Optional
from trainer import train_model
from model_selector import select_best_model
from telegram import send_strategy_update, send_daily_report
from logger import record_result, analyze_daily_log

# 配置日誌
logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

class TrainingMonitor:
    def __init__(self):
        self.train_count = 0
        self.train_success = 0
        self.train_fail = 0

    def record_train(self, success: bool):
        self.train_count += 1
        if success:
            self.train_success += 1
        else:
            self.train_fail += 1

    def get_metrics(self) -> Dict:
        return {
            "train_count": self.train_count,
            "train_success": self.train_success,
            "train_fail": self.train_fail,
            "success_rate": self.train_success / self.train_count if self.train_count > 0 else 0
        }

class TrainingScheduler:
    """Manages periodic training and daily reporting."""
    def __init__(self, train_interval: int = 900, report_time: str = "00:00", timeout: int = 800):
        self.train_interval = train_interval
        self.report_time = report_time
        self.timeout = timeout
        self.monitor = TrainingMonitor()
        self.last_report_file = "last_report.txt"

    def validate_result(self, result: Optional[Dict]) -> bool:
        """Validates training result."""
        required_fields = ["status", "model", "score", "confidence"]
        if not result or any(field not in result for field in required_fields):
            return False
        if result["status"] != "success":
            return False
        if not isinstance(result["score"], (int, float)) or result["score"] < 0:
            return False
        if not isinstance(result["confidence"], (int, float)) or not 0 <= result["confidence"] <= 1:
            return False
        if "tp" in result and "sl" in result:
            if not (isinstance(result["tp"], (int, float)) and isinstance(result["sl"], (int, float))):
                return False
            if not (0 <= result["tp"] <= 100 and 0 <= result["sl"] <= 100):
                return False
        return True

    def should_send_daily_report(self) -> bool:
        """Checks if daily report should be sent."""
        today = date.today()
        if not os.path.exists(self.last_report_file):
            return True
        with open(self.last_report_file, "r") as f:
            last_date = f.read().strip()
        return last_date != str(today)

    def update_last_report_date(self):
        """Updates last report date."""
        with open(self.last_report_file, "w") as f:
            f.write(str(date.today()))

    def job_train(self):
        """Executes model training and strategy update."""
        try:
            logging.info("開始訓練...")
            print("🔁 執行 train_model()...")
            start_time = time.time()
            result = train_model()
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                logging.warning(f"訓練超時: {elapsed:.2f}秒")
                print(f"⚠️ 訓練超時: {elapsed:.2f}秒")
            success = self.validate_result(result)
            self.monitor.record_train(success)
            if not success:
                logging.error("retrain 失敗或結果異常")
                print("❌ retrain 失敗或結果異常")
                return
            logging.info(
                f"retrain 成功: 模型={result['model']}, score={result['score']}, 信心={result['confidence']}"
            )
            print(f"✅ retrain 成功 → 模型：{result['model']}，score：{result['score']}，信心：{result['confidence']}")
            if 'tp' in result and 'sl' in result:
                print(f"🎯 TP：{result['tp']}%，SL：{result['sl']}%")
            record_result(result)
            send_strategy_update(result)
        except Exception as e:
            self.monitor.record_train(False)
            logging.error(f"訓練或推播失敗: {e}")
            print(f"❌ 訓練或推播失敗: {e}")

    def job_daily_report(self):
        """Sends daily report if not sent today."""
        try:
            if self.should_send_daily_report():
                metrics = analyze_daily_log()
                metrics.update(self.monitor.get_metrics())
                send_daily_report(metrics)
                self.update_last_report_date()
                logging.info("已發送日報表")
                print("📊 已發送日報表")
        except Exception as e:
            logging.error(f"日報表發送失敗: {e}")
            print(f"❌ 日報表發送失敗: {e}")

    def run(self):
        """Runs the training and reporting schedule."""
        schedule.every(self.train_interval).seconds.do(self.job_train)
        schedule.every().day.at(self.report_time).do(self.job_daily_report)
        logging.info("訓練排程啟動")
        print("🚀 訓練排程啟動")
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    scheduler = TrainingScheduler()
    scheduler.run()
