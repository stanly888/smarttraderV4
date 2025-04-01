import os, time
from datetime import datetime

while True:
    hour = datetime.now().hour
    if hour < 12:
        os.environ["MODE"] = "train"
        print("🟢 上午 → 實盤資料訓練")
    else:
        os.environ["MODE"] = "backtest"
        print("🟡 下午 → 歷史資料訓練")
    os.system("python main.py")
    time.sleep(3600)
