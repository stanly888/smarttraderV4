import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def push(self, state, action, reward):
        # ✅ 確保 state 一律為 1D numpy array，例如 (10,)
        state = np.array(state).flatten()
        self.add(state, action, reward, next_state=state, done=False)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
