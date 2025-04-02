import requests

def send_strategy_signal(strategy, config):
    token = config["telegram_bot_token"]
    chat_id = config["telegram_chat_id"]
    msg = f"[策略推播]\n方向: {strategy['direction']} 信心: {strategy['confidence']}% TP: {strategy['tp']} SL: {strategy['sl']}"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, data={"chat_id": chat_id, "text": msg})
