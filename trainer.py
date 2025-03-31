# ✅ PPO 強化學習策略 AI：最終優化版（含 TP/SL 槓桿預測 + Reward 強化 + 指標修正）

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

def fetch_ohlcv(symbol='BTC/USDT', timeframe='15m', limit=200):
    exchange = ccxt.okx({
        'enableRateLimit': True,
        'rateLimit': 1000,
    })
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_mfi(df, period=14):
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos_mf = mf.where(tp.diff() > 0, 0).rolling(window=period).sum()
    neg_mf = mf.where(tp.diff() < 0, 0).rolling(window=period).sum()
    mfr = pos_mf / neg_mf
    return 100 - (100 / (1 + mfr))

def add_indicators(df):
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['rsi'] = calculate_rsi(df['close'])
    df['mfi'] = calculate_mfi(df)
    df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
    df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
    return df

class TradingEnv:
    def __init__(self, symbol='BTC/USDT', timeframe='15m', mode='backtest'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.mode = mode
        self.initial_capital = 300
        self.capital = self.initial_capital
        self.index = 30

        self.data = None
        if mode != 'live':
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

        if self.data is None or len(self.data) < 40:
            print("❌ 無法載入足夠的資料，跳過本次訓練。")
            return np.zeros(9)

        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.index]
        return np.array([
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

    def step(self, action):
        self.index += 1
        done = self.index >= len(self.data) - 1
        if done:
            return self._get_state(), 0, True

        current = self.data.iloc[self.index]
        previous = self.data.iloc[self.index - 1]
        change = (current['close'] - previous['close']) / previous['close']
        reward = 0

        if action == 1:  # 做多
            reward = change * 100
            self.capital *= (1 + change)
        elif action == 2:  # 做空
            reward = -change * 100
            self.capital *= (1 - change)

        reward = np.clip(reward, -1, 1)
        return self._get_state(), reward, done

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
            if state is None or (isinstance(state, np.ndarray) and np.all(state == 0)):
                continue

            log_probs, old_log_probs, rewards, actions = [], [], [], []
            done = False
            peak = self.env.capital

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
                peak = max(peak, self.env.capital)
                drawdown = (peak - self.env.capital) / peak
                reward -= drawdown * 0.5

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

            with torch.no_grad():
                probs = self.policy(torch.FloatTensor(state))
            confidence = round(float(probs[final_action].item()) * 100, 2)
            tp = round(2 + confidence * 0.03, 2)
            sl = round(tp / 3, 2)

            leverage = 5
            if confidence > 90:
                leverage = 20
            elif confidence > 75:
                leverage = 10

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

        torch.save(self.policy.state_dict(), "ppo_model.pt")
        print("💾 模型已儲存為 ppo_model.pt")

if __name__ == '__main__':
    def get_mode():
        hour = datetime.now().hour
        return 'live' if hour < 12 else 'backtest'

    trainer = PPOTrainer(symbol='BTC/USDT', timeframe='15m', mode=get_mode())
    trainer.train(episodes=50)
