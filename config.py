import os

# 讀取環境變數
TELEGRAM_TOKEN = os.getenv("telegram_bot_token")
TELEGRAM_CHAT_ID = os.getenv("telegram_chat_id")

BINANCE_API_KEY = os.getenv("binance_api_key")
BINANCE_API_SECRET = os.getenv("binance_api_secret")

# 模擬資本與訓練間隔
INITIAL_CAPITAL = 300
RETRAIN_INTERVAL_MINUTES = 15

# 檔案名稱常數
REWARD_LOG_FILE = "logbook_reward.json"
TRADES_FILE = "real_trades.json"
REPLAY_BUFFER_PATH = "replay_buffer.json"
RETRAIN_STATUS_PATH = "retrain_status.json"