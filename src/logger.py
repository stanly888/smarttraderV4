
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime

def log_strategy_summary(result, config, logbook_path="logbook.json"):
    try:
        if os.path.exists(logbook_path):
            with open(logbook_path, "r") as f:
                data = json.load(f)
        else:
            data = []

        record = {
            "time": datetime.utcnow().isoformat(),
            "model": result["model"],
            "confidence": result["confidence"],
            "leverage": result["leverage"],
            "capital": result["capital"],
            "tp": result["tp"],
            "sl": result["sl"],
            "direction": result["direction"]
        }
        data.append(record)

        with open(logbook_path, "w") as f:
            json.dump(data, f, indent=2)

        plot_performance(data)
    except Exception as e:
        print("Logging error:", e)

def plot_performance(logbook):
    try:
        capitals = [r["capital"] for r in logbook]
        confidences = [r["confidence"] for r in logbook]
        leverages = [r["leverage"] for r in logbook]

        plt.figure()
        plt.plot(capitals, label="Capital")
        plt.title("AI Capital Growth")
        plt.xlabel("Episode")
        plt.ylabel("Capital")
        plt.legend()
        plt.savefig("capital.png")
        plt.close()

        plt.figure()
        plt.plot(confidences, label="Confidence")
        plt.title("Confidence Trend")
        plt.xlabel("Episode")
        plt.ylabel("Confidence")
        plt.legend()
        plt.savefig("confidence.png")
        plt.close()

        plt.figure()
        plt.plot(leverages, label="Leverage")
        plt.title("Leverage Trend")
        plt.xlabel("Episode")
        plt.ylabel("Leverage")
        plt.legend()
        plt.savefig("leverage.png")
        plt.close()
    except Exception as e:
        print("Plotting error:", e)
