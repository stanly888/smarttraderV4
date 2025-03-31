# 日誌記錄模組 (略)
# eval/logbook.py

import os
from datetime import datetime

def log_strategy(strategy, result=None):
    folder = "logs"
    os.makedirs(folder, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")
    filepath = os.path.join(folder, f"{date}.log")

    with open(filepath, "a", encoding="utf-8") as f:
        log = f"[{datetime.now().strftime('%H:%M:%S')}] "
        log += f"{strategy['model']} | {strategy['symbol']} | {strategy['direction']} | "
        log += f"TP: {strategy['tp']} / SL: {strategy['sl']} | 槓桿: {strategy['leverage']} | 信心: {strategy['confidence']}%"
        if result is not None:
            log += f" | 盈虧：{result:.2f}%"
        f.write(log + "\n")
