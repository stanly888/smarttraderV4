import requests
import os
import json
from logger import get_last_signal, record_last_signal

# ✅ 正確讀取 config.json 的方式
with open("config.json", "r") as f:
    config = json.load(f)
TELEGRAM_TOKEN = config["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = config["TELEGRAM_CHAT_ID"]

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

# ✅ 新增每日績效報告推播功能
def send_daily_report(metrics):
    if not metrics:
        print("⚠️ 無法產生每日績效報告（metrics 為空）")
        return

    capital = metrics["final_capital"]
    win_rate = metrics["win_rate"]
    tp_rate = metrics["tp_rate"]
    sl_rate = metrics["sl_rate"]
    avg_conf = metrics["avg_confidence"]
    std_conf = metrics["std_confidence"]
    top_model = metrics["top_model"]
    total_trades = metrics["total_trades"]
    suggest = metrics["recommend_v9"]

    message = f"""
📊 [SmartTrader 每日績效總結]

總交易筆數：{total_trades}
最常使用模型：{top_model}
最終模擬資金：${capital:.2f}

✅ 勝率：{win_rate*100:.1f}%
🎯 TP 命中率：{tp_rate*100:.1f}%
⛔ SL 命中率：{sl_rate*100:.1f}%
🧠 信心平均：{avg_conf:.2f}（波動 ±{std_conf:.2f}）

{"✅ 建議升級至 V9（開始模擬實單）" if suggest else "🔄 尚未達成升級條件，持續觀察"}
"""

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"❌ 每日推播錯誤: {e}")
