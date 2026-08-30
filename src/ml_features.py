"""Set de features técnicas para el clasificador (combina señales débiles de todas las líneas
anteriores: tendencia, momentum, bandas, volatilidad, volumen, canal de breakout).
"""
from __future__ import annotations

import pandas as pd
import ta

FEATURE_COLUMNS = [
    "ema_diff_pct",
    "adx",
    "rsi",
    "bb_position",
    "atr_pct",
    "atr_percentile",
    "volume_ratio",
    "donchian_position",
]


def compute_features(df: pd.DataFrame, atr_percentile_window: int = 720) -> pd.DataFrame:
    out = df.copy()

    ema_fast = ta.trend.EMAIndicator(out["close"], window=50).ema_indicator()
    ema_slow = ta.trend.EMAIndicator(out["close"], window=100).ema_indicator()
    out["ema_diff_pct"] = (ema_fast - ema_slow) / out["close"] * 100

    out["adx"] = ta.trend.ADXIndicator(out["high"], out["low"], out["close"], window=14).adx()
    out["rsi"] = ta.momentum.RSIIndicator(out["close"], window=14).rsi()

    bb = ta.volatility.BollingerBands(out["close"], window=20, window_dev=2.0)
    bb_upper, bb_lower = bb.bollinger_hband(), bb.bollinger_lband()
    out["bb_position"] = (out["close"] - bb_lower) / (bb_upper - bb_lower)

    atr = ta.volatility.AverageTrueRange(out["high"], out["low"], out["close"], window=14).average_true_range()
    out["atr"] = atr
    out["atr_pct"] = atr / out["close"] * 100
    out["atr_percentile"] = out["atr_pct"].rolling(
        window=atr_percentile_window, min_periods=atr_percentile_window // 2
    ).rank(pct=True)

    volume_ma = out["volume"].rolling(window=20).mean()
    out["volume_ratio"] = out["volume"] / volume_ma

    donchian_high = out["high"].rolling(window=20).max()
    donchian_low = out["low"].rolling(window=20).min()
    out["donchian_position"] = (out["close"] - donchian_low) / (donchian_high - donchian_low)

    return out.dropna(subset=FEATURE_COLUMNS + ["atr"])
