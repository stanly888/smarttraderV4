import requests
import json
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

def send_strategy_update(result):
    message = f"""
[SmartTrader 策略更新]
方向：{result['direction']}（信心 {result['confidence']:.2f}）
槓桿：{result['leverage']}x
TP：+{result['tp']}% / SL：-{result['sl']}%
"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    requests.post(url, data=payload)
