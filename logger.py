import json

def record_retrain_status(model_name, reward, confidence):
    data = {
        "model": model_name,
        "score": float(reward),
        "confidence": float(confidence)
    }
    with open("retrain_status.json", "w") as f:
        json.dump(data, f, indent=2)
