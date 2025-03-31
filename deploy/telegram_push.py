import requests
from datetime import datetime

# ✅ Telegram Bot Token & Chat ID
TELEGRAM_TOKEN = "7910051513:AAG7erXokBDz8S5RTChtBuX0PHY69zd0y8o"
TELEGRAM_CHAT_ID = "6579673198"  # ← 已更新這裡

def send_strategy_signal(strategy):
    msg = (
        f"【策略通知】\n"
        f"幣種: {strategy['symbol']}\n"
        f"方向: {strategy['direction']}\n"
        f"信心: {strategy['confidence']}%\n"
        f"槓桿: {strategy['leverage']}x\n"
        f"TP: {strategy['tp']}%\n"
        f"SL: {strategy['sl']}%\n"
        f"理由: {strategy['reason']}"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})

def send_daily_summary(total_trades, win_rate, capital):
    now = datetime.now().strftime('%Y-%m-%d')
    msg = (
        f"📊 <b>SmartTrader AI 每日總結</b>\n"
        f"日期：{now}\n"
        f"交易次數：{total_trades}\n"
        f"勝率：{win_rate:.2f}%\n"
        f"資金：${capital:.2f}"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "HTML"
    })
