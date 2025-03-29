# ✅ PPO 強化學習策略 AI：進階升級（含資金管理、勝率監控）

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

os.makedirs("eval", exist_ok=True)

# ⛏️ 改為 OKX 資料抓取模組
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

# 🔧 技術指標模組（簡化整合）
def add_indicators(df):
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))
    df['mfi'] = 50
    df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
    df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
    return df

# 📦 強化交易環境
class TradingEnv:
    def __init__(self, symbol='BTC/USDT', timeframe='15m', mode='backtest'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.mode = mode
        self.initial_capital = 300
        self.capital = self.initial_capital
        self.index = 30
        self.win_streak = 0
        self.loss_streak = 0
        self.last_reward = 0

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
        self.win_streak = 0
        self.loss_streak = 0
        self.last_reward = 0

        if self.data is None or len(self.data) < 40:
            return np.zeros(9)
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

    def step(self, action):
        if self.index + 1 >= len(self.data):
            return self._get_state(), 0, True

        current = self.data.iloc[self.index]
        next_price = self.data.iloc[self.index + 1]['close']
        now_price = current['close']
        atr = current['atr']
        change = (next_price - now_price) / now_price

        leverage = 5
        risk_fraction = 0.5 if self.loss_streak >= 2 else 1.0  # 動態倉位調整
        trade_capital = self.capital * risk_fraction
        tx_cost = 0.001 * leverage

        reward = 0
        if action == 1:
            reward = (change - tx_cost) * leverage
        elif action == 2:
            reward = (-change - tx_cost) * leverage

        if self.capital < 250:
            reward -= 0.5

        reward = np.clip(reward, -1, 1)

        self.capital *= (1 + reward * risk_fraction)
        self.last_reward = reward

        if reward > 0:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0

        self.index += 1
        done = self.index >= len(self.data) - 2 or self.capital < 10
        return self._get_state(), reward, done

# 其餘 PPO 訓練邏輯保持一致（略）...
