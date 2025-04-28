import pandas as pd
from datetime import datetime
from compute_dual_features import compute_dual_features
from fetch_market_data import fetch_market_data
from ppo_trainer import train_ppo
from a2c_trainer import train_a2c
from dqn_trainer import train_dqn
from logger import record_retrain_status

def train_model(features=None, atr=None, bb_width=None, fib_distance=None):
    """
    - 如果外部傳入 features，直接用
    - 如果沒傳，自己 call compute_dual_features()
    """

    source = "實盤資料"

    try:
        if features is None or atr is None or bb_width is None or fib_distance is None:
            features, (atr, bb_width, fib_distance, volatility_factor) = compute_dual_features("BTC-USDT")
        # 如果外部送進來的是只有前三個，兼容
    except Exception as e:
        print(f"❌ 無法取得實盤資料或計算特徵：{e}")
        return {"status": "error", "message": str(e)}

    # ✅ 檢查特徵有效性
    if features is None or not features.any():
        print("⚠️ 雙週期技術指標異常，略過此次訓練")
        return {"status": "error", "message": "技術指標異常，無有效數據"}

    try:
        # ✅ 呼叫三大模型分別訓練
        result_ppo = train_ppo(features, atr, bb_width, fib_distance)
        result_a2c = train_a2c(features, atr, bb_width, fib_distance)
        result_dqn = train_dqn(features, atr, bb_width, fib_distance)
    except Exception as e:
        print(f"❌ 模型訓練過程中出錯：{e}")
        return {"status": "error", "message": f"模型訓練失敗：{str(e)}"}

    # ✅ 選出當輪最高分模型
    best = max([result_ppo, result_a2c, result_dqn], key=lambda x: x["score"])

    if "model" not in best or "confidence" not in best or "score" not in best:
        return {
            "status": "error",
            "message": "模型訓練結果不完整",
            "raw": best
        }

    # ✅ 記錄 retrain 成功結果
    record_retrain_status(best["model"], best["score"], best["confidence"])

    best["timestamp"] = datetime.utcnow().isoformat()
    best["status"] = "success"
    best["source"] = source

    return best
