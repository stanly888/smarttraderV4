import json
import os
from datetime import datetime

RETRAIN_LOG = "retrain_status.json"
LAST_PUSHED_SIGNAL = "last_signal.json"

def record_result(result):
    with open("logbook.json", "a") as f:
        f.write(json.dumps(result) + "\n")

def analyze_daily_log():
    return {"win_rate": 0.65, "confidence_var": 0.08, "tp_hit": 0.6, "sl_hit": 0.4, "max_drawdown": 0.03}

def record_retrain_status(model_name, reward, confidence):
    data = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reward": reward,
        "confidence": confidence
    }
    with open(RETRAIN_LOG, "w") as f:
        json.dump(data, f)

def get_last_signal():
    if not os.path.exists(LAST_PUSHED_SIGNAL):
        return None
    with open(LAST_PUSHED_SIGNAL, "r") as f:
        return json.load(f)

def record_last_signal(signal):
    with open(LAST_PUSHED_SIGNAL, "w") as f:
        json.dump(signal, f)
