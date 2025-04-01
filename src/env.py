
import numpy as np

class TradingEnv:
    def __init__(self, close, high, low, volume, config):
        self.config = config
        self.initial_capital = config["initial_capital"]
        self.capital = self.initial_capital
        self.index = 0
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.position = 0
        self.history = []

    def reset(self):
        self.capital = self.initial_capital
        self.index = 10  # start from 10 to allow indicators
        self.position = 0
        self.history = []
        return self._get_state()

    def step(self, action, leverage):
        current_price = self.close[self.index]
        next_price = self.close[self.index + 1] if self.index + 1 < len(self.close) else current_price
        reward = 0

        if action == 1:  # Long
            profit = (next_price - current_price) / current_price
        elif action == 2:  # Short
            profit = (current_price - next_price) / current_price
        else:
            profit = 0

        # Adjust with leverage
        adjusted_profit = profit * leverage
        self.capital *= (1 + adjusted_profit)

        # Add penalty for high leverage loss
        if adjusted_profit < 0:
            reward = adjusted_profit - (abs(leverage) / 20) * 0.01
        else:
            reward = adjusted_profit + (abs(leverage) / 20) * 0.01

        self.history.append(self.capital)
        self.index += 1
        done = self.index >= len(self.close) - 2 or self.capital < (1 - self.config["max_drawdown"]) * self.initial_capital
        return self._get_state(), reward, done

    def _get_state(self):
        # Return last 10 returns + indicator values
        if self.index < 10:
            return np.zeros(10)
        window = self.close[self.index-10:self.index]
        returns = np.diff(window) / window[:-1]
        return np.pad(returns, (10 - len(returns), 0), 'constant')
