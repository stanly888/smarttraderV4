import random
import json
import os
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """標準五元組輸入"""
        state = np.array(state).flatten().tolist()
        next_state = np.array(next_state).flatten().tolist()
        self.buffer.append((state, action, reward, next_state, done))

    def push(self, state, action, reward):
        """簡化版本，將 next_state = state 且 done=False"""
        self.add(state, action, reward, next_state=state, done=False)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError(f"ReplayBuffer 不足 {batch_size} 筆樣本")
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
                json.dump(list(self.buffer), f, default=self._convert)
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

    def _convert(self, obj):
        """支援 numpy 轉換為原生型別"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj