import json
from datetime import datetime
import os

STATUS_FILE = "retrain_status.json"

def record_retrain_status(model_name: str, reward: float, confidence: float, source: str = "實盤"):
    """
    記錄每次 retrain 結果（包含時間、模型、信心、得分與資料來源）
    - 若檔案不存在，則自動創建
    - 使用 try-except 捕獲檔案操作錯誤，確保程式穩定運行
    """
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model_name,
        "score": round(float(reward), 4),
        "confidence": round(float(confidence), 4),
        "source": source
    }

    # 確保目錄存在
    try:
        # 確保目錄存在，如果 STATUS_FILE 不是一個目錄的話才創建
        os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    except Exception as e:
        print(f"❌ 無法創建目錄：{e}")

    try:
        # 讀取現有的 retrain_status.json 文件並加載其中的資料
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, "r") as f:
                all_data = json.load(f)
        else:
            all_data = []

        # 將新的 retrain 結果加入到現有資料中
        all_data.append(data)

        # 寫回整個資料清單到檔案
        with open(STATUS_FILE, "w") as f:
            json.dump(all_data, f, indent=2)
        print(f"✅ 已更新 retrain 狀態：{model_name}（score={reward}, confidence={confidence}）")
    except Exception as e:
        # 捕獲檔案操作錯誤並輸出錯誤訊息
        print(f"❌ 無法寫入 retrain_status.json：{e}")
        # 可選：根據需要記錄到日誌或其他位置