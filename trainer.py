from deploy.telegram_push import send_strategy_signal
from eval.logbook import log_strategy
import random, time

class PPOTrainer:
    def __init__(self, symbol, timeframe, mode):
        self.symbol = symbol
        self.timeframe = timeframe
        self.mode = mode

    def train(self, episodes=3):
        for ep in range(episodes):
            capital = round(300 + random.uniform(0, 50), 2)
            direction = random.choice(["做多", "做空", "觀望"])
            confidence = round(random.uniform(65, 95), 2)
            strategy = {
                "symbol": self.symbol,
                "direction": direction,
                "confidence": confidence,
                "leverage": random.choice([5, 10, 20]),
                "tp": round(random.uniform(3, 6), 2),
                "sl": round(random.uniform(1, 3), 2),
                "model": "PPO_Strategy",
                "reason": f"{self.mode} 模式"
            }
            send_strategy_signal(strategy)
            log_strategy(strategy, result=round(confidence - 60, 2))
            print(f"✅ Episode {ep+1} Finished. Capital: {capital}")
            time.sleep(1)
