
import json

def log_trade(result):
    with open("logbook.json", "a") as f:
        f.write(json.dumps(result) + "\n")
