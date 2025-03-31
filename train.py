# train.py ✅ SmartTraderAI 主程式入口點（自動模式切換 + 推播）

from trainer import PPOTrainer
from datetime import datetime
from deploy.telegram_push import send_daily_summary

def get_mode():
    hour = datetime.now().hour
    return 'live' if hour < 12 else 'backtest'

if __name__ == '__main__':
    print("✅ AI 訓練啟動中...")
    mode = get_mode()
    trainer = PPOTrainer(symbol='BTC/USDT', timeframe='15m', mode=mode)
    trainer.train(episodes=50)
    send_daily_summary()
