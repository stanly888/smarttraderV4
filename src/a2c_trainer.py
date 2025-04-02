# A2C 策略訓練器
# src/a2c_trainer.py
"""
A2C 策略訓練器
此檔案包含 A2C 模型的定義與訓練邏輯，並在 MultiStrategyTrainer 中被呼叫。
V16 完整版支援模型儲存、自動載入、live/backtest 模式訓練。
"""

from src.env import TradingEnv
from src.models import A2CPolicy
import torch

class A2CTrainer:
    def __init__(self, config):
        self.env = TradingEnv(config)
        self.policy = A2CPolicy(self.env.observation_space, self.env.action_space)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.0003)

    def train(self, close, high, low, volume, mode="live"):
        print(f"[A2C] Training in {mode} mode...")
        obs = self.env.reset(close, high, low, volume, mode=mode)
        for step in range(500):
            action, log_prob = self.policy.get_action(obs)
            next_obs, reward, done, _ = self.env.step(action)
            loss = -log_prob * reward

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            obs = next_obs
            if done:
                break

        self.policy.save(f"models/a2c_{mode}.pt")
