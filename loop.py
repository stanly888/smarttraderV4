print('SmartTrader Background Worker Running')
# loop.py
import os
import time
import traceback
from datetime import datetime
from src.multi_trainer import MultiStrategyTrainer
from src.utils import load_config, fetch_real_data
from src.logger import log_strategy_summary

INTERVAL_MINUTES = 30

while True:
    try:
        print(f"[{datetime.now()}] Starting training loop...")

        config = load_config()
        mode = "live" if 8 <= datetime.now().hour < 20 else "backtest"
        close, high, low, volume = fetch_real_data(config, mode=mode)

        trainer = MultiStrategyTrainer(config)
        trainer.train_all(close, high, low, volume, mode=mode)

        log_strategy_summary()
        print(f"[{datetime.now()}] Training loop complete. Sleeping {INTERVAL_MINUTES} min...\n")

    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()

    time.sleep(INTERVAL_MINUTES * 60)
