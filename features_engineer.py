import pandas as pd
import numpy as np
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator

def compute_features(df: pd.DataFrame) -> np.ndarray:
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    vwap = (close * volume).cumsum() / volume.cumsum()
    vwap_diff = close - vwap
    obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume().diff()
    mfi = MFIIndicator(high, low, close, volume).money_flow_index()
    atr = AverageTrueRange(high, low, close).average_true_range()
    bb = BollingerBands(close)
    bb_width = bb.bollinger_hband() - bb.bollinger_lband()
    price_pos = (close - bb.bollinger_lband()) / (bb_width + 1e-9)
    ma5, ma10 = SMAIndicator(close, 5).sma_indicator(), SMAIndicator(close, 10).sma_indicator()
    ma_diff = (ma5 - ma10) / (ma10 + 1e-9)
    rsi = RSIIndicator(close).rsi()
    features = np.array([
        vwap_diff.iloc[-1], obv.iloc[-1], mfi.iloc[-1], atr.iloc[-1],
        bb_width.iloc[-1], price_pos.iloc[-1], ma_diff.iloc[-1],
        rsi.iloc[-1], close.pct_change().iloc[-1], volume.pct_change().iloc[-1]
    ])
    return np.nan_to_num((features - features.mean()) / (features.std() + 1e-6))