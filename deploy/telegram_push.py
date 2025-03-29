import requests

TELEGRAM_TOKEN = "7910051513:AAG7erXokBDz8S5RTChtBuX0PHY69zd0y8o"
TELEGRAM_CHAT_ID = "6579673198"

def send_strategy_signal(strategy: dict):
    msg = (
        f"📈 [AI 策略推播]\n"
        f"幣種：{strategy['symbol']}\n"
        f"方向：{strategy['direction']}（信心 {strategy['confidence']}%，槓桿 {strategy['leverage']}x）\n"
        f"模型：{strategy['model']}\n"
        f"理由：{strategy['reason']}\n"
        f"TP：{strategy['tp']}% / SL：{strategy['sl']}%\n"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})