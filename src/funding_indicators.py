"""Alinea el funding rate (eventos cada 8h) sobre velas de 1h, y calcula ATR/volumen para la
estrategia contraria de funding extremo.
"""
from __future__ import annotations

import pandas as pd
import ta

from src.config import FundingConfig
from src.funding_data import fetch_funding_rate


def compute_funding_percentile(funding_rate: pd.Series, lookback_periods: int) -> pd.Series:
    """Percentil rodante del funding rate dentro de su propia ventana histórica reciente."""
    return funding_rate.rolling(window=lookback_periods, min_periods=lookback_periods // 2).rank(pct=True)


def align_funding_to_1h(df_1h: pd.DataFrame, funding_percentile: pd.Series) -> pd.DataFrame:
    """Propaga el percentil de funding (evento discreto cada 8h) sobre el índice de 1h.

    El funding rate reportado en el timestamp T ya es un valor liquidado/conocido en ese momento
    (no una vela en formación como un OHLC diario) — no hace falta un shift adicional, alcanza
    con propagar hacia adelante (ffill) el último valor conocido en cada barra de 1h.
    """
    aligned = funding_percentile.reindex(df_1h.index, method="ffill")
    out = df_1h.copy()
    out["funding_percentile"] = aligned
    return out


def add_funding_indicators(df_1h: pd.DataFrame, symbol: str, cfg: FundingConfig, years: int) -> pd.DataFrame:
    funding_rate = fetch_funding_rate(symbol, years=years)
    funding_percentile = compute_funding_percentile(funding_rate, cfg.lookback_periods)

    out = align_funding_to_1h(df_1h, funding_percentile)
    out["atr"] = ta.volatility.AverageTrueRange(
        out["high"], out["low"], out["close"], window=cfg.atr_period
    ).average_true_range()
    out["volume_ma"] = out["volume"].rolling(window=cfg.volume.volume_ma_period).mean()

    return out.dropna(subset=["funding_percentile", "atr", "volume_ma"])
