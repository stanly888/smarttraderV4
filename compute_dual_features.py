import pandas as pd
import numpy as np
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from fetch_market_data import fetch_market_data

def compute_single_features(df: pd.DataFrame) -> tuple[np.ndarray, float, float, float]:
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    vwap = (close * volume).cumsum() / (volume.cumsum() + 1e-9)
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

    atr_ratio = atr / (close + 1e-9)
    rsi_zone = rsi / 100
    bb_pct = price_pos
    bb_dev = bb_width / (close + 1e-9)
    ma_slope = ma5.diff()

    # 斐波那契支撐壓力
    recent_high = high[-20:].max()
    recent_low = low[-20:].min()
    fib_levels = [recent_high - (recent_high - recent_low) * r for r in [0.236, 0.382, 0.5, 0.618, 0.786]]
    fib_distances = [abs(close.iloc[-1] - level) for level in fib_levels]
    fib_mean_dist = np.mean(fib_distances)

    features = np.array([
        vwap_diff.iloc[-1], obv.iloc[-1], mfi.iloc[-1], atr.iloc[-1],
        bb_width.iloc[-1], price_pos.iloc[-1], ma_diff.iloc[-1],
        rsi.iloc[-1], price_chg.iloc[-1], vol_chg.iloc[-1],
        atr_ratio.iloc[-1], rsi_zone.iloc[-1], bb_pct.iloc[-1],
        bb_dev.iloc[-1], ma_slope.iloc[-1], close.iloc[-1],
        fib_mean_dist  # 第17維：斐波那契距離
    ])

    normalized = np.nan_to_num((features - features.mean()) / (features.std() + 1e-6))
    return normalized, atr.iloc[-1], bb_width.iloc[-1], fib_mean_dist

def compute_dual_features(symbol="BTC-USDT") -> tuple[np.ndarray, tuple[float, float, float]]:
    df_15m = fetch_market_data(symbol=symbol, interval="15m", limit=100)
    df_1h = fetch_market_data(symbol=symbol, interval="1h", limit=100)

    features_15m, atr_15m, bb_15m, fib_15m = compute_single_features(df_15m)
    features_1h, _, _, _ = compute_single_features(df_1h)

    current_price = df_15m["close"].iloc[-1]

    dual_features = np.concatenate([features_15m, features_1h, [current_price]])  # 共 35 維

    # ✅ 注意：回傳 (features, (atr, bb, fib))，是 tuple
    return dual_features, (atr_15m, bb_15m, fib_15m)