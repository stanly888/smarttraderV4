# compute_dual_features.py
import pandas as pd
import numpy as np
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from fetch_market_data import fetch_market_data

def compute_single_features(df: pd.DataFrame) -> tuple[np.ndarray, float]:
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    vwap = (close * volume).cumsum() / volume.cumsum()
    vwap_diff = close - vwap
    obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume().diff()
    mfi = MFIIndicator(high, low, close, volume).money_flow_index()
    atr = AverageTrueRange(high, low, close).average_true_range()
    bb = BollingerBands(close)
    bb_width = bb.bollinger_hband() - bb.bollinger_lband()
    price_pos = (close - bb.bollinger_lband()) / (bb_width + 1e-9)
    ma5 = SMAIndicator(close, 5).sma_indicator()
    ma10 = SMAIndicator(close, 10).sma_indicator()
    ma_diff = (ma5 - ma10) / (ma10 + 1e-9)
    rsi = RSIIndicator(close).rsi()
    price_chg = close.pct_change()
    vol_chg = volume.pct_change()

    # TP/SL 強化特徵
    atr_ratio = atr / close
    rsi_zone = rsi / 100
    bb_pct = price_pos
    bb_dev = bb_width / close
    ma_slope = ma5.diff()

    # 原始 ATR 值（不標準化，留給 TP/SL 使用）
    raw_atr = atr.iloc[-1]

    # ✅ 斐波那契回撤支撐壓力特徵
    recent_high = high[-20:].max()
    recent_low = low[-20:].min()
    fib_levels = [recent_high - (recent_high - recent_low) * r for r in [0.236, 0.382, 0.5, 0.618, 0.786]]
    fib_distances = [abs(close.iloc[-1] - level) for level in fib_levels]
    fib_mean_dist = np.mean(fib_distances)

    # 標準化特徵（用於模型輸入）
    features = np.array([
        vwap_diff.iloc[-1], obv.iloc[-1], mfi.iloc[-1], atr.iloc[-1],
        bb_width.iloc[-1], price_pos.iloc[-1], ma_diff.iloc[-1],
        rsi.iloc[-1], price_chg.iloc[-1], vol_chg.iloc[-1],
        atr_ratio.iloc[-1], rsi_zone.iloc[-1], bb_pct.iloc[-1],
        bb_dev.iloc[-1], ma_slope.iloc[-1], close.iloc[-1],
        fib_mean_dist  # ✅ 第 17 維：斐波那契距離
    ])

    normalized = np.nan_to_num((features - features.mean()) / (features.std() + 1e-6))
    return normalized, raw_atr

def compute_dual_features(symbol="BTC-USDT") -> tuple[np.ndarray, float]:
    df_15m = fetch_market_data(symbol=symbol, interval="15m", limit=100)
    df_1h = fetch_market_data(symbol=symbol, interval="1h", limit=100)

    features_15m, atr_15m = compute_single_features(df_15m)
    features_1h, _ = compute_single_features(df_1h)
    current_price = df_15m["close"].iloc[-1]

    dual_features = np.concatenate([features_15m, features_1h, [current_price]])  # 共 34 維（17 + 16 + 1）
    return dual_features, atr_15m