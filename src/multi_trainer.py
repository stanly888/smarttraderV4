import torch
from torch.distributions import Categorical
from src.models import PPOPolicy, TP_SL_Model
from src.env import TradingEnv
from src.notifications import send_strategy_signal

class MultiStrategyTrainer:
    def __init__(self, config):
        self.config = config
        self.policy = PPOPolicy()
        self.tp_sl = TP_SL_Model()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config["lr_policy"])

    def train_all(self, close, high, low, volume, is_morning=True):
        env = TradingEnv(close, high, low, self.config)
        state = env.reset()
        rewards, actions, log_probs = [], [], []
        done = False
        while not done:
            state_t = torch.FloatTensor(state)
            probs = self.policy(state_t)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            state, reward, done = env.step(action.item())
            rewards.append(reward)
            actions.append(action.item())
            log_probs.append(log_prob)

        strategy = {
            "symbol": "BTC/USDT",
            "direction": ["觀望", "做多", "做空"][actions[-1]],
            "confidence": round(float(probs[action].item()) * 100, 2),
            "tp": 0.02,
            "sl": 0.01,
            "model": "PPO"
        }
        send_strategy_signal(strategy, self.config)
        return strategy

    def backtest(self, close, high, low, volume):
        return self.train_all(close, high, low, volume, is_morning=False)
