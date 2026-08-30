"""Indicadores para la estrategia de breakout (canal de Donchian + volumen)."""
from __future__ import annotations

import pandas as pd
import ta

from src.config import BreakoutConfig


def add_breakout_indicators(df: pd.DataFrame, cfg: BreakoutConfig) -> pd.DataFrame:
    """Añade el canal de Donchian (desplazado 1 vela para no incluirse a sí mismo), ATR y
    media de volumen.
    """
    out = df.copy()

    out["donchian_high"] = out["high"].rolling(window=cfg.donchian.period).max().shift(1)
    out["donchian_low"] = out["low"].rolling(window=cfg.donchian.period).min().shift(1)

    out["atr"] = ta.volatility.AverageTrueRange(
        out["high"], out["low"], out["close"], window=cfg.risk.atr_period
    ).average_true_range()

    out["volume_ma"] = out["volume"].rolling(window=cfg.volume.volume_ma_period).mean()

    return out.dropna()
