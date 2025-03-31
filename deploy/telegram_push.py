# ✅ telegram_push.py：處理 Telegram 推播通知

import requests
import os

# ✅ 替換成你的 TOKEN & CHAT_ID
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "你的TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "你的CHAT_ID")

def send_strategy_signal(strategy):
    text = (
        f"【策略通知】\n"
        f"幣種: {strategy['symbol']}\n"
        f"方向: {strategy['direction']}\n"
        f"信心: {strategy['confidence']}%\n"
        f"槓桿: {strategy['leverage']}x\n"
        f"TP: {strategy['tp']}% / SL: {strategy['sl']}%\n"
        f"理由: {strategy['reason']}"
    )
    send_telegram_message(text)

def send_daily_summary(summary):
    text = (
        f"📊【每日總結】\n"
        f"日期：{summary['date']}\n"
        f"總資金變化：{summary['capital_change']} USDT\n"
        f"勝率：{summary['win_rate']}%\n"
        f"總交易次數：{summary['total_trades']} 次\n"
        f"最佳模型：{summary['best_model']}"
    )
    send_telegram_message(text)

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": text}
    try:
        response = requests.post(url, data=data)
        if not response.ok:
            print("❌ 推播失敗:", response.text)
    except Exception as e:
        print("❌ 推播錯誤:", e)
