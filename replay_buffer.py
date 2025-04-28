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
        state = np.asarray(state, dtype=np.float32).flatten().tolist()
        next_state = np.asarray(next_state, dtype=np.float32).flatten().tolist()
        self.buffer.append((state, action, reward, next_state, done))

    def push(self, state, action, reward):
        """簡化版本，將 next_state = state 且 done=False"""
        self.add(state, action, reward, next_state=state, done=False)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError(f"ReplayBuffer 樣本不足：目前 {len(self.buffer)} 筆，需要 {batch_size} 筆")
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        # ✅ 統一轉為 float32
        states = states.astype(np.float32)
        next_states = next_states.astype(np.float32)
        rewards = rewards.astype(np.float32)
        actions = actions.astype(np.int64)
        dones = dones.astype(np.bool_)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def save(self, path="replay_buffer.json"):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)  # ✅ 自動建立資料夾
            with open(path, "w") as f:
                json.dump(list(self.buffer), f, default=self._convert_safe, allow_nan=False)
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

    def _convert_safe(self, obj):
        """支援 numpy 型別轉換，且保證safe"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            val = float(obj)
            if np.isnan(val) or np.isinf(val):
                return 0.0
            return val
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        return obj
