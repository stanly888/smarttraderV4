import csv
from datetime import datetime

def log_strategy(strategy: dict, result: float):
    with open("eval/strategy_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            strategy["symbol"],
            strategy["direction"],
            strategy["confidence"],
            strategy["leverage"],
            strategy["tp"],
            strategy["sl"],
            strategy["model"],
            strategy["reason"],
            result
        ])
