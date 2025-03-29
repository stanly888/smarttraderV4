import requests

TELEGRAM_TOKEN = "7910051513:AAG7erXokBDz8S5RTChtBuX0PHY69zd0y8o"
TELEGRAM_CHAT_ID = "6579673198"

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    except Exception as e:
        print("Telegram 推播失敗：", e)

def send_strategy_signal(strategy: dict):
    text = f"[SmartTrader AI 推播]\n策略：{strategy['model']}（{strategy['reason']}）\n方向：{strategy['direction']}\n信心：{strategy['confidence']}%\n槓桿：{strategy['leverage']}x\nTP：{strategy['tp']}% / SL：{strategy['sl']}%"
    send_telegram(text)

def send_daily_summary(trade_count, win_count, final_capital):
    message = f"""📊 AI 每日訓練總結

🔁 今日進場次數：{trade_count} 次
✅ 勝率：{round(win_count / trade_count * 100, 2)}%
💰 初始資金：300 USDT
📈 訓練後資金：{round(final_capital, 2)} USDT
"""
    send_telegram(message)
