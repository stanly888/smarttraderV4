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
        import statistics

def analyze_daily_log():
    if not os.path.exists("logbook.json"):
        return {}

    with open("logbook.json", "r") as f:
        lines = f.readlines()

    results = [json.loads(line) for line in lines if line.strip()]
    if not results:
        return {}

    capital = 300
    capital_curve = []
    wins = 0
    losses = 0
    tp_hits = 0
    sl_hits = 0
    confidence_list = []
    model_usage = {}

    for r in results:
        reward = r.get("reward", 0)
        capital += reward
        capital_curve.append(capital)

        if reward > 0:
            wins += 1
        else:
            losses += 1

        if r.get("hit_tp"):
            tp_hits += 1
        if r.get("hit_sl"):
            sl_hits += 1

        conf = r.get("confidence")
        if conf:
            confidence_list.append(conf)

        model = r.get("model", "Unknown")
        model_usage[model] = model_usage.get(model, 0) + 1

    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    tp_rate = tp_hits / (wins + losses) if (wins + losses) > 0 else 0
    sl_rate = sl_hits / (wins + losses) if (wins + losses) > 0 else 0
    avg_conf = statistics.mean(confidence_list) if confidence_list else 0
    std_conf = statistics.stdev(confidence_list) if len(confidence_list) > 1 else 0
    top_model = max(model_usage.items(), key=lambda x: x[1])[0]

    return {
        "final_capital": capital,
        "capital_curve": capital_curve,
        "win_rate": win_rate,
        "tp_rate": tp_rate,
        "sl_rate": sl_rate,
        "avg_confidence": avg_conf,
        "std_confidence": std_conf,
        "top_model": top_model,
        "total_trades": len(results),
        "recommend_v9": win_rate > 0.65 and len(results) >= 10
    }
        
