
import torch
import os
from src.trainer import PPOTrainer
from src.a2c_trainer import A2CTrainer

class MultiStrategyTrainer:
    def __init__(self, close, high, low, config):
        self.close = close
        self.high = high
        self.low = low
        self.config = config

    def train_all(self, close, high, low, is_morning):
        ppo = PPOTrainer(self.close, self.high, self.low, self.config)
        a2c = A2CTrainer(self.close, self.high, self.low, self.config)

        ppo_result = ppo.train(close, high, low, is_morning)
        a2c_result = a2c.train(close, high, low, is_morning)

        best = max([ppo_result, a2c_result], key=lambda x: x['score'])

        model_name = best['model']
        model_object = best['model_object']
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
        torch.save(model_object.state_dict(), f"saved_models/{model_name}_best.pt")

        return best
