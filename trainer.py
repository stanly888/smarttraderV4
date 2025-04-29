import pandas as pd
from datetime import datetime
from compute_dual_features import compute_dual_features
from fetch_market_data import fetch_market_data
from ppo_trainer import train_ppo
from a2c_trainer import train_a2c
from dqn_trainer import train_dqn
from logger import record_retrain_status

def train_model(features=None, atr=None, bb_width=None, fib_distance=None, volatility_factor=None):
    """
    - 如果外部傳入 features，直接使用。
    - 如果未傳入，透過 `compute_dual_features()` 計算雙週期特徵。
    """

    source = "實盤資料"

    try:
        # 如果外部缺少任何參數，則使用 compute_dual_features 計算特徵
        if features is None or atr is None or bb_width is None or fib_distance is None or volatility_factor is None:
            features, (atr, bb_width, fib_distance, volatility_factor) = compute_dual_features("BTC-USDT")
    except Exception as e:
        print(f"❌ 無法取得實盤資料或計算特徵：{e}")
        return {"status": "error", "message": str(e)}

    # 檢查特徵的有效性
    if features is None or not features.any():
        print("⚠️ 雙週期技術指標異常，略過此次訓練")
        return {"status": "error", "message": "技術指標異常，無有效數據"}

    try:
        # 呼叫三大模型分別訓練
        result_ppo = train_ppo(features, atr, bb_width, fib_distance, volatility_factor)
        result_a2c = train_a2c(features, atr, bb_width, fib_distance, volatility_factor)
        result_dqn = train_dqn(features, atr, bb_width, fib_distance, volatility_factor)
    except Exception as e:
        print(f"❌ 模型訓練過程中出錯：{e}")
        return {"status": "error", "message": f"模型訓練失敗：{str(e)}"}

    # 選擇當輪最高分的模型
    best = max([result_ppo, result_a2c, result_dqn], key=lambda x: x["score"])

    # 檢查是否有完整的模型結果
    if "model" not in best or "confidence" not in best or "score" not in best:
        return {
            "status": "error",
            "message": "模型訓練結果不完整",
            "raw": best
        }

    # 記錄成功的 retrain 結果
    record_retrain_status(best["model"], best["score"], best["confidence"])

    # 增加其他元資料
    best["timestamp"] = datetime.utcnow().isoformat()
    best["status"] = "success"
    best["source"] = source

    return best