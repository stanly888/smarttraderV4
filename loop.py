
import time
from trainer import train_and_predict
from telegram import send_message

def main():
    while True:
        result = train_and_predict()
        message = (
            "[SmartTrader AI 策略更新]\n"
            f"方向：{result['action']}\n"
            f"信心：{result['confidence']:.2f}\n"
            f"TP：{result['tp']:.2f}% / SL：{result['sl']:.2f}%"
        )
        send_message(message)
        time.sleep(1800)

if __name__ == "__main__":
    main()
