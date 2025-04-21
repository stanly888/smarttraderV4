import random
import json
import os
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """添加完整五元組（RL 標準格式）"""
        self.buffer.append((state, action, reward, next_state, done))

    def push(self, state, action, reward):
        """簡化版本，會將 next_state 與 done 固定為當前狀態與 False"""
        state = np.array(state).flatten().tolist()
        self.add(state, action, reward, next_state=state, done=False)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer too small to sample {batch_size} entries")
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def save(self, path="replay_buffer.json"):
        try:
            with open(path, "w") as f:
                json.dump(list(self.buffer), f)
            print(f"✅ Replay Buffer 已儲存：{path}")
        except Exception as e:
            print(f"❌ 儲存 Replay Buffer 失敗：{e}")

    def load(self, path="replay_buffer.json"):
        if not os.path.exists(path):
            print(f"⚠️ 找不到 Replay Buffer 檔案：{path}")
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.buffer = deque(data, maxlen=self.buffer.maxlen)
            print(f"✅ Replay Buffer 已載入：{path}")
        except Exception as e:
            print(f"❌ 載入 Replay Buffer 失敗：{e}")