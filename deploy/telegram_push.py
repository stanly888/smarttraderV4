import requests

TELEGRAM_TOKEN = "7910051513:AAG7erXokBDz8S5RTChtBuX0PHY69zd0y8o"
TELEGRAM_CHAT_ID = "6579673198"

def send_strategy_signal(strategy):
    message = (
        f"[SmartTrader AI 推播]\n"
        f"策略：{strategy['model']}（{strategy['reason']}）\n"
        f"方向：{strategy['direction']}\n"
        f"信心：{strategy['confidence']}%\n"
        f"槓桿：{strategy['leverage']}x\n"
        f"TP：{strategy['tp']}% / SL：{strategy['sl']}%"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"❌ Telegram 推播失敗：{e}")
