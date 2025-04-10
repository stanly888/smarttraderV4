
import time
from telegram import send_message
from trainer import train_model

def main():
    while True:
        result = train_model()
        send_message(f"[SmartTrader] 策略已更新：信心 {result['confidence']:.2f}")
        time.sleep(1800)

if __name__ == "__main__":
    main()
