import json, os
from datetime import datetime

def log_strategy_summary(result, config):
    log_path = "logbook.json"
    log = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log = json.load(f)
    result["timestamp"] = str(datetime.now())
    log.append(result)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
