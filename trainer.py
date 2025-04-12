from ppo_trainer import train_ppo
from a2c_trainer import train_a2c
from logger import record_retrain_status

def train_model():
    print("🔁 開始 retrain 模型...")

    # 執行 PPO 與 A2C 模型訓練
    result_ppo = train_ppo()
    result_ppo["model"] = "PPO"  # 加入模型標記

    result_a2c = train_a2c()
    result_a2c["model"] = "A2C"  # 加入模型標記

    # Debug log 各自績效
    print(f"📊 PPO 模型：score={result_ppo.get('score')}, confidence={result_ppo.get('confidence')}")
    print(f"📊 A2C 模型：score={result_a2c.get('score')}, confidence={result_a2c.get('confidence')}")

    # 比較 score，選出最強模型
    best_result = max([result_ppo, result_a2c], key=lambda x: x["score"])
    print(f"✅ 最強模型：{best_result['model']}，score={best_result['score']}")

    # 寫入 retrain 狀態檔
    try:
        record_retrain_status(
            model_name=best_result["model"],
            reward=best_result["score"],
            confidence=best_result["confidence"]
        )
        print(f"📝 已記錄 retrain_status.json（模型：{best_result['model']}）")
    except Exception as e:
        print(f"❌ 寫入 retrain_status.json 失敗：{e}")

    return best_result
