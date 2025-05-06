import requests
import pandas as pd
import time

def fetch_market_data(symbol: str = "BTC-USDT", interval: str = "15m", limit: int = 100, retries: int = 3) -> pd.DataFrame:
    """
    從 BingX API 取得市場 K 線資料並返回為 DataFrame
    :param symbol: 交易對名稱（例如：'BTC-USDT'）
    :param interval: 時間間隔（例如：'15m'、'1h'、'1d'等）
    :param limit: 取得的資料筆數（預設為100）
    :param retries: 最大重試次數
    :return: 包含 K 線數據的 DataFrame
    """
    url = "https://open-api.bingx.com/openApi/spot/v1/market/kline"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    for attempt in range(retries):
        try:
            # 發送請求至 BingX API
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()  # 檢查回應狀態碼

            data = response.json()
            
            # 檢查回應是否包含預期的數據
            if data.get("code") != 0 or "data" not in data:
                raise ValueError(f"BingX 回傳異常：{data}")

            raw = data["data"]
            
            # 解析原始數據並建立 DataFrame
            df = pd.DataFrame(raw, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "quoteVolume", "count"
            ])

            # 將 timestamp 轉換為 datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            # 選擇需要的欄位並轉換數據類型
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            
            # 返回按時間排序的資料
            return df.sort_values("timestamp").reset_index(drop=True)

        except requests.exceptions.RequestException as e:
            # 網絡請求錯誤處理
            print(f"❌ 網絡請求失敗: {e}. 嘗試 {attempt + 1}/{retries} 次")
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)  # 指數退避機制
        except ValueError as e:
            # 解析數據錯誤處理
            print(f"❌ 解析 BingX 回傳的數據失敗: {e}")
            raise
        except Exception as e:
            # 其他錯誤處理
            print(f"❌ 獲取 BingX K 線資料時發生錯誤: {e}")
            raise
