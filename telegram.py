import requests
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
