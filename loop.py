import os
import json
import time
import traceback
from datetime import datetime
from src.multi_trainer import MultiStrategyTrainer
from src.utils import fetch_real_data
from src.notifications import send_strategy_signal

def load_config():
    try:
        with open("config/config.json") as f:
            config = json.load(f)
    except Exception as e:
        raise RuntimeError(f"❌ 讀取 config/config.json 失敗: {e}")

    for key in ["binance_api_key", "binance_api_secret", "telegram_bot_token", "telegram_chat_id"]:
        config[key] = os.getenv(key, config.get(key))
    return config

def logbook_save(entry):
    try:
        filename = "logbook.json"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.append(entry)
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"❌ logbook 寫入失敗: {e}")

def main_loop():
    while True:
        print(f"\n🕒 Retrain 開始 - {datetime.now().isoformat()}")
        try:
            config = load_config()
            close, high, low, volume = fetch_real_data(config)
            trainer = MultiStrategyTrainer(config)
            result = trainer.train_all(close, high, low, volume)

            print("✅ 訓練完成:", result)
            logbook_save(result)

            # 發送推播
            send_strategy_signal({
                "symbol": result.get("symbol", "BTC/USDT"),
                "direction": result.get("direction", "觀望"),
                "reason": "SmartTrader 自動訓練推播",
                "leverage": result.get("leverage", 5),
                "confidence": result.get("confidence", 95),
                "tp": result.get("tp", 0.03),
                "sl": result.get("sl", 0.015),
                "model": result.get("model", "SmartTrader_V10.1")
            }, config)

            print("✅ Telegram 推播已送出")

        except Exception as e:
            print(f"❌ retrain 發生錯誤：{e}")
            traceback.print_exc()

        print("⏳ 等待下一次 retrain...\n")
        time.sleep(3600)  # 每小時 retrain 一次

if __name__ == "__main__":
    main_loop()# loop logic will go here
import os
import json
import time
import traceback
from datetime import datetime
from multi_trainer import MultiStrategyTrainer
from utils import fetch_real_data
from notifications import send_strategy_signal

def load_config():
    try:
        with open("config/config.json") as f:
            config = json.load(f)
    except Exception as e:
        raise RuntimeError(f"❌ 讀取 config/config.json 失敗: {e}")

    for key in ["binance_api_key", "binance_api_secret", "telegram_bot_token", "telegram_chat_id"]:
        config[key] = os.getenv(key, config.get(key))
    return config

def logbook_save(entry):
    try:
        filename = "logbook.json"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.append(entry)
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"❌ logbook 寫入失敗: {e}")

def main_loop():
    while True:
        print(f"\n🕒 Retrain 開始 - {datetime.now().isoformat()}")
        try:
            config = load_config()
            close, high, low, volume = fetch_real_data(config)
            trainer = MultiStrategyTrainer(config)
            result = trainer.train_all(close, high, low, volume)

            print("✅ 訓練完成:", result)
            logbook_save(result)

            # 發送推播
            send_strategy_signal({
                "symbol": result.get("symbol", "BTC/USDT"),
                "direction": result.get("direction", "觀望"),
                "reason": "SmartTrader 自動訓練推播",
                "leverage": result.get("leverage", 5),
                "confidence": result.get("confidence", 95),
                "tp": result.get("tp", 0.03),
                "sl": result.get("sl", 0.015),
                "model": result.get("model", "SmartTrader_V10.1")
            }, config)

            print("✅ Telegram 推播已送出")

        except Exception as e:
            print(f"❌ retrain 發生錯誤：{e}")
            traceback.print_exc()

        print("⏳ 等待下一次 retrain...\n")
        time.sleep(3600)  # 每小時 retrain 一次

if __name__ == "__main__":
    main_loop()
