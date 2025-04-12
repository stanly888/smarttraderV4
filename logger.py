import json

def record_result(result):
    with open("logbook.json", "a") as f:
        f.write(json.dumps(result) + "\n")

def analyze_daily_log():
    return {"win_rate": 0.65, "confidence_var": 0.08, "tp_hit": 0.6, "sl_hit": 0.4, "max_drawdown": 0.03}