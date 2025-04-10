
import requests
import json

def send_message(text):
    with open("config.json", "r") as f:
        config = json.load(f)
    token = config["bot_token"]
    chat_id = config["chat_id"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, json=payload)
