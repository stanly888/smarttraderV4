import numpy as np

class TradingEnv:
    def __init__(self, close, high, low, config):
        self.close = close
        self.high = high
        self.low = low
        self.config = config
        self.index = 0
        self.capital = config["initial_capital"]
        self.position = 0

    def reset(self):
        self.index = 0
        self.capital = self.config["initial_capital"]
        self.position = 0
        return self._get_state()

    def step(self, action):
        reward = 0
        if action == 1: self.position = 1
        elif action == 2: self.position = -1
        else: self.position = 0
        self.index += 1
        done = self.index >= len(self.close) - 1
        return self._get_state(), reward, done

    def _get_state(self):
        start = max(0, self.index - 10)
        return np.array(self.close[start:self.index+1])
