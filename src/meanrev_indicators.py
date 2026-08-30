"""Indicadores para la estrategia de reversión a la media (Bollinger + RSI + volumen)."""
from __future__ import annotations

import pandas as pd
import ta

from src.config import MeanReversionConfig


def add_meanrev_indicators(df: pd.DataFrame, cfg: MeanReversionConfig) -> pd.DataFrame:
    """Añade bandas de Bollinger (basis/upper/lower), RSI, ATR y media de volumen."""
    out = df.copy()

    bb = ta.volatility.BollingerBands(
        out["close"], window=cfg.bollinger.period, window_dev=cfg.bollinger.num_std
    )
    out["bb_basis"] = bb.bollinger_mavg()
    out["bb_upper"] = bb.bollinger_hband()
    out["bb_lower"] = bb.bollinger_lband()

    out["rsi"] = ta.momentum.RSIIndicator(out["close"], window=cfg.rsi.period).rsi()

    out["atr"] = ta.volatility.AverageTrueRange(
        out["high"], out["low"], out["close"], window=cfg.risk.atr_period
    ).average_true_range()

    out["volume_ma"] = out["volume"].rolling(window=cfg.volume.volume_ma_period).mean()

    return out.dropna()
