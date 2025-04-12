from ppo_trainer import train_ppo
from a2c_trainer import train_a2c
from logger import record_retrain_status

def train_model():
    # 執行 PPO 與 A2C 模型訓練
    result_ppo = train_ppo()
    result_ppo["model"] = "PPO"  # 加入模型標記

    result_a2c = train_a2c()
    result_a2c["model"] = "A2C"  # 加入模型標記

    # 比較兩個模型的 score（績效）選出最強者
    best_result = max([result_ppo, result_a2c], key=lambda x: x["score"])

    # 記錄 retrain 狀態（給推播與日誌使用）
    record_retrain_status(
        model_name=best_result["model"],
        reward=best_result["score"],
        confidence=best_result["confidence"]
    )

    # 回傳這筆策略
    return best_result
