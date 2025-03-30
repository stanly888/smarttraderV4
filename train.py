# 主程式：訓練啟動
from trainer import PPOTrainer

if __name__ == '__main__':
    trainer = PPOTrainer()
    trainer.train(episodes=50)
