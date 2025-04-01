
import requests

def send_strategy_signal(strategy, config):
    token = config["telegram_bot_token"]
    chat_id = config["telegram_chat_id"]
    message = (
        f"【AI 交易策略推播】\n"
        f"交易對：{strategy['symbol']}\n"
        f"方向：{strategy['direction']}\n"
        f"信心：{strategy['confidence']}%\n"
        f"槓桿：{strategy['leverage']}x\n"
        f"TP：{strategy['tp']} / SL：{strategy['sl']}\n"
        f"模式：{'實盤' if strategy['is_morning'] else '歷史'}\n"
        f"資金：{strategy['capital']}\n"
        f"模型：{strategy['model']}"
    )
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"Telegram 推播失敗: {response.text}")
    except Exception as e:
        print("Telegram Error:", e)
