print('SmartTrader Main Running')
# main.py
from src.multi_trainer import MultiStrategyTrainer
from src.utils import fetch_real_data, load_config
from src.logger import log_strategy_summary

if __name__ == "__main__":
    print("SmartTrader Main Running")

    config = load_config()
    close, high, low, volume = fetch_real_data(config)

    trainer = MultiStrategyTrainer(config)
    trainer.train_all(close, high, low, volume)
