import requests
import os
import json
from logger import get_last_signal, record_last_signal

# 從 config.json 讀取 Telegram Token 和 Chat ID
with open("config.json", "r") as f:
    cfg = json.load(f)

TELEGRAM_TOKEN = cfg["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = cfg["TELEGRAM_CHAT_ID"]

def send_strategy_update(result):
    # 準備目前訊號
    current = {
        "model": result['model'],
        "direction": result['direction'],
        "confidence": result['confidence'],
        "leverage": result['leverage'],
        "tp": result['tp'],
        "sl": result['sl']
    }

    # 避免重複推播（如果跟上次一模一樣，就略過）
    last = get_last_signal()
    if last == current:
        print("⚠️ 重複策略訊號，略過推播")
        return

    # 記錄這次訊號
    record_last_signal(current)

    # 讀取 retrain 訓練資訊
    retrain_info = {}
    if os.path.exists("retrain_status.json"):
        with open("retrain_status.json", "r") as f:
            retrain_info = json.load(f)

    # 建立推播訊息
    message = f"""
📡 [SmartTrader 策略推播]
模型：{current['model']}（更新於：{retrain_info.get('timestamp', 'N/A') }）
方向：{current['direction']}（信心：{current['confidence']:.2f}）
槓桿：{current['leverage']}x
TP：+{current['tp']}% / SL：-{current['sl']}%
"""

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("✅ 策略訊號推播成功")
        else:
            print(f"❌ 推播失敗: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ 推播錯誤: {e}")import requests
import os
import json
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from logger import get_last_signal, record_last_signal

def send_strategy_update(result):
    last = get_last_signal()
    current = {
        "model": result['model'],
        "direction": result['direction'],
        "confidence": result['confidence'],
        "leverage": result['leverage'],
        "tp": result['tp'],
        "sl": result['sl']
    }

    if last == current:
        print("⚠️ 重複策略，略過推播")
        return

    record_last_signal(current)

    retrain_info = {}
    if os.path.exists("retrain_status.json"):
        with open("retrain_status.json", "r") as f:
            retrain_info = json.load(f)

    message = f"""
📡 [SmartTrader 策略推播]
模型：{current['model']}（更新於：{retrain_info.get('timestamp', 'N/A') }）
方向：{current['direction']}（信心：{current['confidence']:.2f}）
槓桿：{current['leverage']}x
TP：+{current['tp']}% / SL：-{current['sl']}%
"""

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    requests.post(url, data=payload)
