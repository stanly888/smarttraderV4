
import random

def train_and_predict():
    # 模擬 AI 訓練 + 推理
    confidence = random.uniform(0.7, 0.95)
    tp = random.uniform(1.5, 3.5)
    sl = random.uniform(0.8, 2.0)
    action = random.choice(["Long", "Short", "Skip"])
    return {"confidence": confidence, "tp": tp, "sl": sl, "action": action}
