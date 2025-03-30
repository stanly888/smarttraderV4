
from trainer import PPOTrainer
from datetime import datetime

def get_mode():
    hour = datetime.now().hour
    return 'live' if hour < 12 else 'backtest'

if __name__ == '__main__':
    trainer = PPOTrainer(mode=get_mode())
    trainer.train(episodes=50)
