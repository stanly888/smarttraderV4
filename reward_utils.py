# reward_utils.py
import numpy as np

def simulate_reward(direction: str, tp: float, sl: float, leverage: float, fib_distance: float) -> float:
    """
    模擬 TP/SL reward，加入 fib_distance 偏好與極值懲罰
    """
    # 模擬 TP 命中機率
    hit = np.random.rand()
    raw = tp if hit < 0.5 else -sl

    # 交易成本
    fee = 0.0004 * leverage * 2
    funding = 0.00025 * leverage

    # 斐波那契懲罰（偏離 0.618 越遠，reward 越低）
    fib_penalty = abs(fib_distance - 0.618)

    # 懲罰：TP 太誇張（>0.2）或 SL 太小（<0.002）
    tp_penalty = 0.3 if tp > 0.2 else 0
    sl_penalty = 0.5 if sl < 0.002 else 0

    base_reward = raw * leverage - fee - funding
    adjusted = base_reward * (1 - fib_penalty) - tp_penalty - sl_penalty

    return round(adjusted, 4)
