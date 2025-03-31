
# ✅ SmartTrader AI V7：整合 TP/SL 預測 + Reward 強化 + 每日績效紀錄

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from deploy.telegram_push import send_strategy_signal, send_daily_summary
from eval.logbook import log_strategy
from datetime import datetime
import random
import json

class TradingEnv:
    def __init__(self):
        self.capital = 300
        self.index = 0
        self.data = None

    def reset(self):
        self.index = 0
        self.capital = 300
        return np.random.randn(9)

    def step(self, action):
        self.capital *= (1 + np.random.uniform(-0.01, 0.01))
        reward = np.random.uniform(-1, 1)
        self.index += 1
        done = self.index >= 50
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

class TP_SL_Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.model(x)

class PPOTrainer:
    def __init__(self):
        self.env = TradingEnv()
        self.policy = PPOPolicy(9, 3)
        self.tp_sl_model = TP_SL_Model(9)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.tp_sl_optimizer = optim.Adam(self.tp_sl_model.parameters(), lr=1e-3)
        self.old_policy = PPOPolicy(9, 3)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.daily_summary = []

    def train(self, episodes=50):
        for ep in range(episodes):
            state = self.env.reset()
            log_probs, old_log_probs, rewards, actions = [], [], [], []
            states = []
            done = False
            while not done:
                state_tensor = torch.FloatTensor(state)
                states.append(state_tensor)

                probs = self.policy(state_tensor)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                old_probs = self.old_policy(state_tensor)
                old_log_prob = Categorical(old_probs).log_prob(action)

                state, reward, done = self.env.step(action.item())
                reward -= abs(reward) * 0.1

                log_probs.append(log_prob)
                old_log_probs.append(old_log_prob)
                rewards.append(reward)
                actions.append(action.item())

            self.old_policy.load_state_dict(self.policy.state_dict())

            returns, G = [], 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns)

            ratios = torch.exp(torch.stack(log_probs) - torch.stack(old_log_probs))
            surr1 = ratios * returns
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * returns
            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # TP/SL 預測訓練
            state_batch = torch.stack(states)
            targets = torch.tensor([[2 + random.random() * 2, 1 + random.random()] for _ in states])
            preds = self.tp_sl_model(state_batch)
            tp_sl_loss = nn.MSELoss()(preds, targets)
            self.tp_sl_optimizer.zero_grad()
            tp_sl_loss.backward()
            self.tp_sl_optimizer.step()

            final_action = actions[-1]
            direction = ['觀望', '做多', '做空'][final_action]
            with torch.no_grad():
                probs = self.policy(torch.FloatTensor(state))
                confidence = round(float(probs[final_action].item()) * 100, 2)
                tp_sl = self.tp_sl_model(torch.FloatTensor(state))
                tp = round(float(tp_sl[0].item()), 2)
                sl = round(float(tp_sl[1].item()), 2)

            leverage = 5
            if confidence > 90: leverage = 20
            elif confidence > 75: leverage = 10

            strategy = {
                'symbol': 'BTC/USDT',
                'direction': direction,
                'reason': f'AI PPO 策略（live 模式）',
                'leverage': leverage,
                'confidence': confidence,
                'tp': tp,
                'sl': sl,
                'model': 'SmartTrader_V7'
            }

            self.daily_summary.append(strategy)
            send_strategy_signal(strategy)
            log_strategy(strategy, result=round((self.env.capital - 300) / 3, 2))
            print(f"✅ Episode {ep+1} | Capital: {round(self.env.capital, 2)}")

        if datetime.now().hour == 23:
            send_daily_summary(self.daily_summary)

# 主程式略
