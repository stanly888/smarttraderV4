import os
import pickle
import random

MODEL_PATH = "models/best_model.pkl"

def train_model(env, model=None, episodes=30):
    """
    簡化訓練邏輯：模擬訓練後輸出報酬、勝率、信心等資訊
    """
    total_reward = 0
    wins = 0
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action = random.choice(env.action_space)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if reward > 0:
                wins += 1

    capital = env.total_profit
    win_rate = (wins / episodes) * 100
    confidence = min(1.0, win_rate / 100)

    return {
        "model": "mock_model",  # 你可以換成 PPO/A2C 模型物件
        "capital": capital,
        "win_rate": win_rate,
        "confidence": confidence
    }

def save_model(model):
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

def load_model_if_exists():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None
