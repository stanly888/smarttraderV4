import random
import json
import os
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=1000):
        """初始化回放緩衝區，設定最大容量"""
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """添加五元組（state, action, reward, next_state, done）至緩衝區"""
        state = np.asarray(state, dtype=np.float32).flatten().tolist()  # 轉換為 1D 列表
        next_state = np.asarray(next_state, dtype=np.float32).flatten().tolist()
        self.buffer.append((state, action, reward, next_state, done))

    def push(self, state, action, reward):
        """簡化版的 add 方法，假設 next_state = state 且 done=False"""
        self.add(state, action, reward, next_state=state, done=False)

    def sample(self, batch_size):
        """從緩衝區隨機取出一批樣本"""
        if len(self.buffer) < batch_size:
            raise ValueError(f"ReplayBuffer 樣本不足：目前 {len(self.buffer)} 筆，需要 {batch_size} 筆")
        batch = random.sample(self.buffer, batch_size)  # 隨機取樣
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        # 確保轉換為適當的數據類型
        states = states.astype(np.float32)
        next_states = next_states.astype(np.float32)
        rewards = rewards.astype(np.float32)
        actions = actions.astype(np.int64)
        dones = dones.astype(np.bool_)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """返回緩衝區的大小"""
        return len(self.buffer)

    def size(self):
        """返回緩衝區的大小（與 __len__ 相同）"""
        return len(self.buffer)

    def clear(self):
        """清空緩衝區"""
        self.buffer.clear()

    def save(self, path="replay_buffer.json"):
        """將回放緩衝區保存為 JSON 檔案"""
        # 新增防呆：若 path 是空的，則設為預設值
        if not path:
            path = "replay_buffer.json"
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)  # 自動創建資料夾
            with open(path, "w") as f:
                json.dump(list(self.buffer), f, default=self._convert_safe, allow_nan=False)
            print(f"✅ Replay Buffer 已儲存：{path}")
        except Exception as e:
            print(f"❌ 儲存 Replay Buffer 失敗：{e}")

    def load(self, path="replay_buffer.json"):
        """從 JSON 檔案載入回放緩衝區"""
        if not path:
            path = "replay_buffer.json"
        if not os.path.exists(path):
            print(f"⚠️ 找不到 Replay Buffer 檔案：{path}")
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.buffer = deque(data, maxlen=self.buffer.maxlen)  # 載入資料並限制最大長度
            print(f"✅ Replay Buffer 已載入：{path}")
        except Exception as e:
            print(f"❌ 載入 Replay Buffer 失敗：{e}")

    def _convert_safe(self, obj):
        """支援將 numpy 型別轉換為 Python 原生型別，並確保數據安全"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            val = float(obj)
            if np.isnan(val) or np.isinf(val):
                return 0.0  # 防止 NaN 或 Inf
            return val
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        return obj
