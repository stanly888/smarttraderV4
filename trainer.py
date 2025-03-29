# ✅ PPO 強化學習策略 AI：進階升級（含模型儲存與讀取）

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from deploy.telegram_push import send_strategy_signal
from eval.logbook import log_strategy
import random
import time
from datetime import datetime
import os
import ccxt

# ⛏️ 資料抓取模組（整合進 trainer.py）
def fetch_ohlcv(symbol='BTC/USDT', timeframe='15m', limit=200):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

# 🔧 技術指標模組（簡化整合）
def add_indicators(df):
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))
    df['mfi'] = 50  # 假設常數，未實作完整
    df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
    df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
    return df

# 🧠 PPO 策略模型
class PPOPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

# 📦 強化交易環境
class TradingEnv:
    def __init__(self, symbol='BTC/USDT', timeframe='15m', mode='backtest'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.mode = mode
        self.initial_capital = 300
        self.capital = self.initial_capital
        self.index = 30

        if mode == 'live':
            self.data = None
        else:
            self.data = add_indicators(fetch_ohlcv(symbol, timeframe, 200))

    def reset(self):
        if self.mode == 'live':
            for _ in range(3):
                try:
                    self.data = add_indicators(fetch_ohlcv(self.symbol, self.timeframe, 200))
                    break
                except Exception as e:
                    print(f"Error fetching live data: {e}")
                    time.sleep(5)
            else:
                print("Falling back to empty data")
                self.data = pd.DataFrame()
        self.index = 30
        self.capital = self.initial_capital
        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.index]
        state = np.array([
            row['ma5'] / row['close'] - 1,
            row['ma10'] / row['close'] - 1,
            row['obv'] / self.data['obv'].std(),
            row['mfi'] / 100,
            row['vwap'] / row['close'] - 1,
            row['rsi'] / 100,
            row['atr'] / row['close'],
            (row['bb_upper'] - row['close']) / row['close'],
            (row['bb_lower'] - row['close']) / row['close']
        ])
        return state

    def step(self, action, leverage=1):
        current = self.data.iloc[self.index]
        next_price = self.data.iloc[self.index + 1]['close']
        now_price = current['close']
        change = (next_price - now_price) / now_price
        tx_cost = 0.001 * leverage

        reward = 0
        if action == 1:
            reward = (change - tx_cost) * leverage
        elif action == 2:
            reward = (-change - tx_cost) * leverage

        reward = np.clip(reward, -1, 1)
        self.capital *= (1 + reward)
        self.index += 1
        done = self.index >= len(self.data) - 2 or self.capital < 10
        return self._get_state(), reward, done

# 🚀 PPO 強化訓練器
class PPOTrainer:
    def __init__(self, symbol='BTC/USDT', timeframe='15m', mode='backtest'):
        self.env = TradingEnv(symbol, timeframe, mode)
        self.policy = PPOPolicy(input_dim=9, output_dim=3)
        self.old_policy = PPOPolicy(input_dim=9, output_dim=3)

        if os.path.exists("ppo_model.pt"):
            self.policy.load_state_dict(torch.load("ppo_model.pt"))
            print("✅ 已載入現有模型 ppo_model.pt")

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

                state_next, reward, done = self.env.step(action.item())
                log_probs.append(log_prob)
                old_log_probs.append(old_log_prob)
                rewards.append(reward)
                actions.append(action.item())
                state = state_next

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
            confidence = round(min(max(returns[-1].item()*100, 60), 99), 2)

            if direction == '觀望':
                leverage = 1
            elif confidence > 90:
                leverage = 20
            elif confidence > 70:
                leverage = 10
            else:
                leverage = 5

            strategy = {
                'symbol': self.symbol,
                'direction': direction,
                'reason': f'AI PPO 策略（{self.env.mode} 模式）',
                'leverage': leverage,
                'confidence': confidence,
                'tp': round(random.uniform(3, 6), 2),
                'sl': round(random.uniform(1, 3), 2),
                'model': 'PPO_Strategy'
            }
            send_strategy_signal(strategy)
            log_strategy(strategy, result=round((self.env.capital - 300) / 3, 2))
            print(f"✅ Episode {ep+1} Finished. Capital: {round(self.env.capital, 2)}")

        torch.save(self.policy.state_dict(), "ppo_model.pt")
        print("💾 模型已儲存為 ppo_model.pt")

if __name__ == '__main__':
    def get_mode():
        hour = datetime.now().hour
        return 'live' if hour < 12 else 'backtest'

    trainer = PPOTrainer(symbol='BTC/USDT', timeframe='15m', mode=get_mode())
    trainer.train(episodes=50)
