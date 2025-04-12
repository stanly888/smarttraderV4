import json
from datetime import datetime
import pytz
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Callable, Optional, List
from ppo_trainer import train_ppo
from a2c_trainer import train_a2c
from logger import record_retrain_status

def validate_result(result: Optional[Dict], model_name: str) -> bool:
    """
    Validates the training result to ensure it contains required fields.

    Args:
        result: The result dictionary from training.
        model_name: Name of the model (e.g., 'PPO', 'A2C').

    Returns:
        bool: True if valid, False otherwise.
    """
    if not result:
        print(f"❌ {model_name} training returned None")
        return False
    if "score" not in result or "confidence" not in result:
        print(f"❌ {model_name} result missing 'score' or 'confidence'")
        return False
    if not isinstance(result["score"], (int, float)) or not isinstance(result["confidence"], (int, float)):
        print(f"❌ {model_name} result has invalid types for 'score' or 'confidence'")
        return False
    return True

def safe_record_retrain_status(
    model_name: str,
    reward: float,
    confidence: float,
    primary_file: str = "retrain_status.json",
    backup_file: str = "retrain_status_backup.json",
    retries: int = 3
) -> bool:
    """
    Safely records retrain status to a file with retry and backup mechanisms.

    Args:
        model_name: Name of the model.
        reward: Training reward/score.
        confidence: Model confidence.
        primary_file: Primary file to write status.
        backup_file: Backup file if primary write fails.
        retries: Number of retry attempts.

    Returns:
        bool: True if successful, False otherwise.
    """
    status_data = {
        "model": model_name,
        "reward": reward,
        "confidence": confidence,
        "timestamp": datetime.now(pytz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    for attempt in range(retries):
        try:
            record_retrain_status(model_name=model_name, reward=reward, confidence=confidence)
            print(f"📝 Successfully logged to {primary_file} (model: {model_name})")
            return True
        except Exception as e:
            print(f"❌ Failed to write {primary_file} (attempt {attempt + 1}/{retries}): {e}")
            if attempt == retries - 1:
                try:
                    with open(backup_file, "a", encoding="utf-8") as f:
                        json.dump(status_data, f, ensure_ascii=False)
                        f.write("\n")
                    print(f"📝 Wrote to backup file {backup_file}")
                    return True
                except Exception as backup_e:
                    print(f"❌ Failed to write backup file {backup_file}: {backup_e}")
                    return False
    return False

def train_single_model(model_name: str, train_fn: Callable) -> Optional[Dict]:
    """
    Trains a single model and validates its result.

    Args:
        model_name: Name of the model.
        train_fn: Function to train the model.

    Returns:
        Optional[Dict]: Training result if valid, None otherwise.
    """
    try:
        print(f"🔄 Training {model_name}...")
        result = train_fn()
        if validate_result(result, model_name):
            result["model"] = model_name
            print(f"📊 {model_name}: score={result['score']}, confidence={result['confidence']}")
            return result
        return None
    except Exception as e:
        print(f"❌ {model_name} training error: {e}")
        return None

def select_best_model(results: List[Dict]) -> Optional[Dict]:
    """
    Selects the best model based on score.

    Args:
        results: List of valid training results.

    Returns:
        Optional[Dict]: Best result or None if no valid results.
    """
    if not results:
        print("❌ No valid training results")
        return None
    best_result = max(results, key=lambda x: x["score"])
    print(f"✅ Best model: {best_result['model']}, score={best_result['score']}")
    return best_result

def train_model(max_workers: int = 2) -> Dict:
    """
    Trains multiple reinforcement learning models, selects the best based on score,
    logs the result, and returns it with metadata.

    Args:
        max_workers: Maximum number of parallel training workers.

    Returns:
        Dict: Best model result with keys 'model', 'score', 'confidence', 'timestamp',
              or error info if training fails.
    """
    print("🔁 Starting model retraining...")

    # Define trainers
    trainers = {
        "PPO": train_ppo,
        "A2C": train_a2c,
        # Add more models here in the future
    }

    results = []
    # Parallel training
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {model: executor.submit(train_single_model, model, fn) for model, fn in trainers.items()}
        for model_name, future in futures.items():
            result = future.result()
            if result:
                results.append(result)

    # Select best model
    best_result = select_best_model(results)
    if not best_result:
        return {"status": "error", "message": "All models failed to train"}

    # Log retrain status
    safe_record_retrain_status(
        model_name=best_result["model"],
        reward=best_result["score"],
        confidence=best_result["confidence"]
    )

    # Add timestamp and status
    best_result["timestamp"] = datetime.now(pytz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    best_result["status"] = "success"

    return best_result

if __name__ == "__main__":
    result = train_model()
    print(f"🏁 Training completed: {result}")
