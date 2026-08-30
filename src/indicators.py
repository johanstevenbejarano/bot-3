"""Cálculo de los indicadores técnicos usados por las 3 capas de la estrategia (largo y corto)."""
from __future__ import annotations

import pandas as pd
import ta

from src.config import StrategyConfig


def add_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Añade EMA rápida/lenta, ADX, ATR, media de volumen, swing low y swing high."""
    out = df.copy()

    out["ema_fast"] = ta.trend.EMAIndicator(out["close"], window=cfg.trend.ema_fast).ema_indicator()
    out["ema_slow"] = ta.trend.EMAIndicator(out["close"], window=cfg.trend.ema_slow).ema_indicator()
    out["adx"] = ta.trend.ADXIndicator(
        out["high"], out["low"], out["close"], window=cfg.trend.adx_period
    ).adx()

    out["atr"] = ta.volatility.AverageTrueRange(
        out["high"], out["low"], out["close"], window=cfg.risk.atr_period
    ).average_true_range()

    out["volume_ma"] = out["volume"].rolling(window=cfg.volume.volume_ma_period).mean()
    out["swing_low"] = out["low"].rolling(window=cfg.risk.swing_lookback).min()
    out["swing_high"] = out["high"].rolling(window=cfg.risk.swing_lookback).max()

    # Percentil de volatilidad relativa (ATR/precio) dentro de su propia ventana histórica
    # reciente — usado para el filtro de régimen en strategy.compute_layers. Es dimensional
    # (fracción 0-1), no necesita reescalarse para FractionalBacktest.
    atr_pct = out["atr"] / out["close"]
    out["atr_percentile"] = atr_pct.rolling(
        window=cfg.regime.lookback_bars, min_periods=cfg.regime.lookback_bars // 2
    ).rank(pct=True)

    return out.dropna()
