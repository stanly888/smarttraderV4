
import time
from trainer import train_model
from telegram import send_strategy_update

while True:
    result = train_model()
    send_strategy_update(result)
    time.sleep(900)  # 每 15 分鐘 retrain 一次
