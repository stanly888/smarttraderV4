# trainer.py
import pandas as pd
from datetime import datetime
from compute_dual_features import compute_dual_features  # ✅ 回傳 features + atr
from fetch_market_data import fetch_market_data
from ppo_trainer import train_ppo
from a2c_trainer import train_a2c
from dqn_trainer import train_dqn
from logger import record_retrain_status

def train_model():
    source = "實盤資料"

    try:
        # ✅ 抓取雙週期特徵（33維）與 ATR 值
        features, atr = compute_dual_features("BTC-USDT")
    except Exception as e:
        print(f"❌ 無法取得實盤資料或計算特徵：{e}")
        return {"status": "error", "message": str(e)}

    # ✅ 檢查特徵有效性
    if features is None or not features.any():
        print("⚠️ 雙週期技術指標異常，略過此次訓練")
        return {"status": "error", "message": "技術指標異常，無有效數據"}

    # ✅ 傳入 features 與 atr 給各模型訓練
    result_ppo = train_ppo(features, atr)
    result_a2c = train_a2c(features, atr)
    result_dqn = train_dqn(features, atr)

    # ✅ 選出最佳模型
    best = max([result_ppo, result_a2c, result_dqn], key=lambda x: x["score"])

    if "model" not in best or "confidence" not in best or "score" not in best:
        return {
            "status": "error",
            "message": "模型訓練結果不完整",
            "raw": best
        }

    # ✅ 記錄結果
    record_retrain_status(best["model"], best["score"], best["confidence"])

    best["timestamp"] = datetime.utcnow().isoformat()
    best["status"] = "success"
    best["source"] = source
    return best