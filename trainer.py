from ppo_trainer import train_ppo
from a2c_trainer import train_a2c
from logger import record_retrain_status

def train_model():
    result_ppo = train_ppo()
    result_ppo["model"] = "PPO"

    result_a2c = train_a2c()
    result_a2c["model"] = "A2C"

    # 比較 reward / score，選擇最強者
    best_result = max([result_ppo, result_a2c], key=lambda x: x["score"])
    
    # 寫入 retrain_status.json
    record_retrain_status(
        model_name=best_result["model"],
        reward=best_result["score"],
        confidence=best_result["confidence"]
    )

    return best_result
