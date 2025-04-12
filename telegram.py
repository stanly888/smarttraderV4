import requests
import json

# 從 JSON 檔載入 Telegram Token 與 Chat ID
with open("config.json", "r") as f:
    config = json.load(f)
TELEGRAM_TOKEN = config["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = config["TELEGRAM_CHAT_ID"]

def send_strategy_update(result):
    message = f"""
📡 [SmartTrader 策略推播]
模型：{result.get('model')}
方向：{result['direction']}（信心：{result['confidence']:.2f}）
槓桿：{result['leverage']}x
TP：+{result['tp']}% / SL：-{result['sl']}%
"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    requests.post(url, data=payload)

def send_daily_report(metrics):
    message = f"""
📊 [SmartTrader 每日績效報告]
📈 勝率：{metrics['win_rate'] * 100:.1f}%
📊 信心波動：±{metrics['confidence_var']:.2f}
🎯 TP 命中率：{metrics['tp_hit'] * 100:.1f}% / SL 命中率：{metrics['sl_hit'] * 100:.1f}%
📉 最大回撤：{metrics['max_drawdown'] * 100:.1f}%

📌 升級建議：{"✅ 建議進入 V9 (模擬實單)" if metrics['win_rate'] >= 0.65 and metrics['max_drawdown'] < 0.05 else "🧪 繼續觀察"}
"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    requests.post(url, data=payload)

