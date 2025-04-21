import json
import requests
from datetime import datetime
from pytz import timezone

# ✅ 安全讀取 config
try:
    with open("config.json", "r") as f:
        config = json.load(f)
    TELEGRAM_TOKEN = config["TELEGRAM_TOKEN"]
    TELEGRAM_CHAT_ID = config["TELEGRAM_CHAT_ID"]
except Exception as e:
    TELEGRAM_TOKEN = ""
    TELEGRAM_CHAT_ID = ""
    print(f"❌ 無法載入 Telegram 設定：{e}")

def send_strategy_update(result):
    raw_timestamp = result.get('timestamp')
    try:
        utc_dt = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
        taipei_time = utc_dt.astimezone(timezone("Asia/Taipei"))
        formatted_time = taipei_time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        formatted_time = raw_timestamp or "N/A"

    message = f"""📡 <b>[SmartTrader 策略推播]</b>
模型：{result['model']}（更新於：{formatted_time}）
方向：{result.get('direction', 'N/A')}（信心：{result.get('confidence', 0):.2f}）
槓桿：{result.get('leverage', 'N/A')}x
TP：+{result.get('tp', 0)}% / SL：-{result.get('sl', 0)}%
"""
    send_telegram_message(message)

def send_daily_report(metrics):
    message = f"""📊 <b>[SmartTrader 每日績效總結]</b>
總交易筆數：{metrics.get("total_trades", 0)}
最常使用模型：{metrics.get("top_model", 'N/A')}
最終模擬資金：${metrics.get("final_capital", 0):.2f}

✅ 勝率：{metrics.get("win_rate", 0)*100:.1f}%
🎯 TP 命中率：{metrics.get("tp_rate", 0)*100:.1f}%
⛔ SL 命中率：{metrics.get("sl_rate", 0)*100:.1f}%
🧠 信心平均：{metrics.get("avg_confidence", 0):.2f}（波動 ±{metrics.get("std_confidence", 0):.2f}）

{"✅ 建議升級至 V9" if metrics.get("recommend_v9") else "🔄 尚未達成升級條件"}
"""
    send_telegram_message(message)

def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ 未設定 Telegram Token 或 Chat ID，無法推播")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }

    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        print("✅ 推播成功")
    except Exception as e:
        print(f"❌ Telegram 推播錯誤：{e} | 回應：{response.text}")