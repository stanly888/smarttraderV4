# ✅ 已修正路徑與匯入錯誤，同時整合每日推播與 TP/SL 平均統計

from trainer import PPOTrainer  # 正確：train.py 與 trainer.py 在同一層
from datetime import datetime


def get_mode():
    hour = datetime.now().hour
    return 'live' if hour < 12 else 'backtest'


if __name__ == '__main__':
    trainer = PPOTrainer(symbol='BTC/USDT', timeframe='15m', mode=get_mode())
    trainer.train(episodes=50)
    trainer.send_daily_summary()  # ✅ 新增：每天訓練後自動推播總結
