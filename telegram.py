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

    # 如果跟上一筆一模一樣，略過推播
    if last == current:
        print("⚠️ 推播內容重複，略過此次推送。")
        return

    # 紀錄目前這筆訊號
    record_last_signal(current)

    # 讀取 retrain 訓練時間
    retrain_info = {}
    if os.path.exists("retrain_status.json"):
        with open("retrain_status.json", "r") as f:
            retrain_info = json.load(f)

    # 組裝推播訊息
    message = f"""
📡 [SmartTrader 策略推播]
模型：{current['model']}（更新於：{retrain_info.get('timestamp', 'N/A') }）
方向：{current['direction']}（信心：{current['confidence']:.2f}）
槓桿：{current['leverage']}x
TP：+{current['tp']}% / SL：-{current['sl']}%
"""

    # 發送 Telegram 訊息
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"❌ 推播錯誤: {e}")
