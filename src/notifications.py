import requests
import os

def send_telegram_message(text, config):
    token = config["telegram_bot_token"]
    chat_id = config["telegram_chat_id"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"[Telegram] 發送失敗: {response.text}")
    except Exception as e:
        print(f"[Telegram] 發送錯誤: {e}")

def send_strategy_signal(strategy, config):
    token = config["telegram_bot_token"]
    chat_id = config["telegram_chat_id"]
    message = (
        f"【交易信號】\n"
        f"幣種: {strategy['symbol']}\n"
        f"方向: {strategy['direction']}\n"
        f"信心: {strategy['confidence']}%\n"
        f"槓桿: {strategy['leverage']}x\n"
        f"TP: {strategy['tp']} | SL: {strategy['sl']}\n"
        f"策略: {strategy['model']}\n"
        f"理由: {strategy['reason']}"
    )
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"[Telegram] 策略推播失敗: {response.text}")
    except Exception as e:
        print(f"[Telegram] 策略推播錯誤: {e}")

def send_daily_summary(summary_list, config):
    token = config["telegram_bot_token"]
    chat_id = config["telegram_chat_id"]

    if not summary_list:
        return

    summary = "【每日策略總結】\n"
    for strat in summary_list:
        summary += (
            f"{strat['symbol']} - {strat['direction']} "
            f"(信心: {strat['confidence']}%, TP: {strat['tp']}, SL: {strat['sl']})\n"
        )

    # 先發送圖（如果存在）
    if os.path.exists("output/capital_curve.png"):
        photo_url = f"https://api.telegram.org/bot{token}/sendPhoto"
        try:
            with open("output/capital_curve.png", "rb") as photo:
                files = {"photo": photo}
                data = {"chat_id": chat_id}
                requests.post(photo_url, data=data, files=files)
        except Exception as e:
            print(f"[Telegram] 發送資金圖失敗: {e}")

    # 再發送摘要文字
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": summary}
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"[Telegram] 總結推播失敗: {response.text}")
    except Exception as e:
        print(f"[Telegram] 總結推播錯誤: {e}")
