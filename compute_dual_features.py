import pandas as pd
import numpy as np
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from fetch_market_data import fetch_market_data

def compute_single_features(df: pd.DataFrame) -> tuple[np.ndarray, float, float, float, float]:
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    # 計算 VWAP 及其差異
    vwap = (close * volume).cumsum() / (volume.cumsum() + 1e-9)
    vwap_diff = close - vwap
    
    # 計算 OBV (On-Balance Volume)
    obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume().diff()
    
    # 計算 MFI (Money Flow Index)
    mfi = MFIIndicator(high, low, close, volume).money_flow_index()
    
    # 計算 ATR (Average True Range)
    atr = AverageTrueRange(high, low, close).average_true_range()
    
    # 計算布林帶 (Bollinger Bands) 及其寬度
    bb = BollingerBands(close)
    bb_width = bb.bollinger_hband() - bb.bollinger_lband()
    price_pos = (close - bb.bollinger_lband()) / (bb_width + 1e-9)
    
    # 計算簡單移動平均線 (SMA)
    ma5 = SMAIndicator(close, 5).sma_indicator()
    ma10 = SMAIndicator(close, 10).sma_indicator()
    ma_diff = (ma5 - ma10) / (ma10 + 1e-9)
    
    # 計算 RSI (Relative Strength Index)
    rsi = RSIIndicator(close).rsi()
    
    # 計算價格變動率與成交量變動率
    price_chg = close.pct_change()
    vol_chg = volume.pct_change()

    # 計算 ATR 比率、RSI 區間、布林帶比率與布林帶寬度比率
    atr_ratio = atr / (close + 1e-9)
    rsi_zone = rsi / 100
    bb_pct = price_pos
    bb_dev = bb_width / (close + 1e-9)
    ma_slope = ma5.diff()

    # 斐波那契支撐壓力位
    recent_high = high[-20:].max()
    recent_low = low[-20:].min()
    fib_levels = [recent_high - (recent_high - recent_low) * r for r in [0.236, 0.382, 0.5, 0.618, 0.786]]
    fib_distances = [abs(close.iloc[-1] - level) for level in fib_levels]
    fib_mean_dist = np.mean(fib_distances)

    # 計算波動指標：用 ATR 比例與 BB 寬比例
    atr_pct = atr.iloc[-1] / close.iloc[-1]
    bb_pct_value = bb_width.iloc[-1] / close.iloc[-1]

    # 設定 volatility_factor：波動越大，factor越大
    volatility_factor = (atr_pct + bb_pct_value) * 10  # 可以自己調倍率
    volatility_factor = np.clip(volatility_factor, 0.5, 2.0)  # 限制在0.5x～2.0x之間

    features = np.array([
        vwap_diff.iloc[-1], obv.iloc[-1], mfi.iloc[-1], atr.iloc[-1],
        bb_width.iloc[-1], price_pos.iloc[-1], ma_diff.iloc[-1],
        rsi.iloc[-1], price_chg.iloc[-1], vol_chg.iloc[-1],
        atr_ratio.iloc[-1], rsi_zone.iloc[-1], bb_pct.iloc[-1],
        bb_dev.iloc[-1], ma_slope.iloc[-1], close.iloc[-1],
        fib_mean_dist
    ])

    # 特徵標準化
    normalized = np.nan_to_num((features - np.mean(features)) / (np.std(features) + 1e-6))

    # 返回標準化特徵、ATR、BB寬度、Fib距離和波動因子
    return normalized, atr.iloc[-1], bb_width.iloc[-1], fib_mean_dist, volatility_factor

def compute_dual_features(symbol="BTC-USDT") -> tuple[np.ndarray, tuple[float, float, float, float]]:
    # 取得 15 分鐘與 1 小時的資料
    df_15m = fetch_market_data(symbol=symbol, interval="15m", limit=100)
    df_1h = fetch_market_data(symbol=symbol, interval="1h", limit=100)

    # 計算 15 分鐘和 1 小時的特徵
    features_15m, atr_15m, bb_15m, fib_15m, vol_factor_15m = compute_single_features(df_15m)
    features_1h, _, _, _, _ = compute_single_features(df_1h)

    # 當前價格
    current_price = df_15m["close"].iloc[-1]

    # 合併 15 分鐘與 1 小時的特徵，並包含當前價格
    dual_features = np.concatenate([features_15m, features_1h, [current_price]])  # 34+1 = 35維

    return dual_features, (atr_15m, bb_15m, fib_15m, vol_factor_15m)
