import time
from src.utils import load_config, fetch_real_data
from src.models import train_model, save_model, load_model_if_exists
from src.logger import log_metrics
from src.notifications import send_telegram_message
from src.env import TradingEnv

print("SmartTrader Background Worker Running...")

try:
    # === 載入設定 ===
    config = load_config()
    mode = "live"  # or "backtest"

    # === 載入歷史資料 ===
    candles = fetch_real_data(config, mode=mode)

    if not candles or len(candles) < 50:
        raise ValueError("歷史 K 線資料不足，無法訓練")

    # === 建立交易環境 ===
    env = TradingEnv(candles)

    # === 載入舊模型（如果有）===
    model = load_model_if_exists()

    # === 開始訓練 ===
    result = train_model(env, model=model, episodes=30)

    # === 儲存模型 ===
    save_model(result["model"])

    # === 記錄訓練數據到 logbook ===
    log_metrics(result)

    # === 發送 Telegram 推播 ===
    message = (
        f"✅ SmartTrader V16 訓練完成\n"
        f"模式：{mode}\n"
        f"資金報酬：{result['capital']:.2f}\n"
        f"勝率：{result['win_rate']:.2f}%\n"
        f"信心分數：{result['confidence']:.2f}"
    )
    send_telegram_message(config, message)

except Exception as e:
    print("❌ 執行失敗：", e)
    try:
        send_telegram_message(config, f"❌ SmartTrader 執行錯誤：{str(e)}")
    except:
        pass
