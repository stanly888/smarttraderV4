
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from deploy.telegram_push import send_strategy_signal
from eval.logbook import log_strategy

class TradingEnv:
    def __init__(self, symbol='BTC/USDT', timeframe='15m', mode='backtest'):
        self.mode = mode
        self.capital = 300
        self.index = 0
        self.data = self._mock_data()

    def _mock_data(self):
        return None

    def reset(self):
        self.index = 0
        return np.random.randn(9)

    def step(self, action):
        self.capital *= (1 + np.random.uniform(-0.01, 0.01))
        reward = np.random.uniform(-1, 1)
        done = self.index >= 49
        self.index += 1
        return np.random.randn(9), reward, done

class PPOPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class PPOTrainer:
    def __init__(self, symbol='BTC/USDT', timeframe='15m', mode='backtest'):
        self.env = TradingEnv(symbol, timeframe, mode)
        self.policy = PPOPolicy(9, 3)
        self.old_policy = PPOPolicy(9, 3)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.symbol = symbol
        self.eps_clip = 0.2

    def train(self, episodes=50):
        for ep in range(episodes):
            state = self.env.reset()
            log_probs, old_log_probs, rewards, actions = [], [], [], []
            done = False
            while not done:
                state_tensor = torch.FloatTensor(state)
                with torch.no_grad():
                    old_probs = self.old_policy(state_tensor)
                probs = self.policy(state_tensor)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                old_log_prob = Categorical(old_probs).log_prob(action)
                state, reward, done = self.env.step(action.item())
                log_probs.append(log_prob)
                old_log_probs.append(old_log_prob)
                rewards.append(reward)
                actions.append(action.item())
            self.old_policy.load_state_dict(self.policy.state_dict())
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns)
            log_probs = torch.stack(log_probs)
            old_log_probs = torch.stack(old_log_probs)
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * returns
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * returns
            loss = -torch.min(surr1, surr2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
           final_action = actions[-1]
direction = ['觀望', '做多', '做空'][final_action]

# ✅ 信心分數：使用模型輸出的機率
with torch.no_grad():
    probs = self.policy(torch.FloatTensor(state))
confidence = round(float(probs[final_action].item()) * 100, 2)

# ✅ TP / SL 動態調整
tp = round(2 + confidence * 0.03, 2)
sl = round(tp / 3, 2)

# ✅ 槓桿根據信心分數調整
if confidence > 90:
    leverage = 20
elif confidence > 70:
    leverage = 10
else:
    leverage = 5

# ✅ 組合策略訊息
strategy = {
    'symbol': self.symbol,
    'direction': direction,
    'reason': f'AI PPO 策略（{self.env.mode} 模式）',
    'leverage': leverage,
    'confidence': confidence,
    'tp': tp,
    'sl': sl,
    'model': 'PPO_Strategy'
}

send_strategy_signal(strategy)
log_strategy(strategy, result=round((self.env.capital - 300) / 3, 2))
print(f"✅ Episode {ep+1} Finished. Capital: {round(self.env.capital, 2)}")

