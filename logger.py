import json

def record_retrain_status(model_name, reward, confidence):
    with open("retrain_status.json", "w") as f:
        json.dump({"model": model_name, "score": reward, "confidence": confidence}, f)