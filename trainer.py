import pandas as pd
from datetime import datetime
from compute_dual_features import compute_dual_features
from fetch_market_data import fetch_market_data
from ppo_trainer import train_ppo
from a2c_trainer import train_a2c
from dqn_trainer import train_dqn
from logger import record_retrain_status

def train_model(features=None, atr=None, bb_width=None, fib_distance=None, volatility_factor=None):
    """
    - If external features are passed, use them directly.
    - If not, compute dual features internally using `compute_dual_features()`.
    """

    source = "實盤資料"

    try:
        # If features or some inputs are missing, calculate them using compute_dual_features
        if features is None or atr is None or bb_width is None or fib_distance is None or volatility_factor is None:
            features, (atr, bb_width, fib_distance, volatility_factor) = compute_dual_features("BTC-USDT")
    except Exception as e:
        print(f"❌ Failed to fetch real-time data or compute features: {e}")
        return {"status": "error", "message": str(e)}

    # Check the validity of the features
    if features is None or not features.any():
        print("⚠️ Invalid dual-cycle technical indicators, skipping this training")
        return {"status": "error", "message": "Invalid technical indicators, no valid data"}

    try:
        # Train the three models
        result_ppo = train_ppo(features, atr, bb_width, fib_distance, volatility_factor)
        result_a2c = train_a2c(features, atr, bb_width, fib_distance, volatility_factor)
        result_dqn = train_dqn(features, atr, bb_width, fib_distance, volatility_factor)
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return {"status": "error", "message": f"Model training failed: {str(e)}"}

    # Select the best model based on score
    best = max([result_ppo, result_a2c, result_dqn], key=lambda x: x["score"])

    if "model" not in best or "confidence" not in best or "score" not in best:
        return {
            "status": "error",
            "message": "Model training result is incomplete",
            "raw": best
        }

    # Record the successful retraining result
    record_retrain_status(best["model"], best["score"], best["confidence"])

    # Add additional metadata
    best["timestamp"] = datetime.utcnow().isoformat()
    best["status"] = "success"
    best["source"] = source

    return best
