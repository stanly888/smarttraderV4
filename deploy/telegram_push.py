import requests

TELEGRAM_TOKEN = "你的 Telegram Bot Token"
TELEGRAM_CHAT_ID = "你的 Chat ID"

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
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})

