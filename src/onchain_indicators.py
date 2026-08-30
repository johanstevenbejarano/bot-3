"""Percentil rodante de direcciones activas on-chain (evento diario) y su alineación sobre velas
de 1h -- mismo criterio de alineación que `funding_indicators.py`/`macro_indicators.py` (ffill
del último valor conocido, sin mirar al futuro).
"""
from __future__ import annotations

import pandas as pd
import ta

from src.config import OnchainConfig


def compute_activity_percentile(active_addresses: pd.Series, lookback_periods: int) -> pd.Series:
    return active_addresses.rolling(window=lookback_periods, min_periods=lookback_periods // 2).rank(pct=True)


def align_activity_to_1h(df_1h: pd.DataFrame, activity_percentile: pd.Series) -> pd.DataFrame:
    aligned = activity_percentile.reindex(df_1h.index, method="ffill")
    out = df_1h.copy()
    out["activity_percentile"] = aligned
    return out


def add_onchain_indicators(df_1h: pd.DataFrame, symbol: str, cfg: OnchainConfig) -> pd.DataFrame:
    """Pipeline completo: trae direcciones activas (BTC o ETH según `symbol`), calcula su
    percentil rodante, lo alinea sobre el índice de 1h del precio, y añade ATR/volumen."""
    from src.onchain_data import fetch_active_addresses

    active_addresses = fetch_active_addresses(symbol)
    activity_percentile = compute_activity_percentile(active_addresses, cfg.lookback_periods)
    out = align_activity_to_1h(df_1h, activity_percentile)

    out["atr"] = ta.volatility.AverageTrueRange(
        out["high"], out["low"], out["close"], window=cfg.atr_period
    ).average_true_range()
    out["volume_ma"] = out["volume"].rolling(window=cfg.volume.volume_ma_period).mean()

    return out.dropna(subset=["activity_percentile", "atr", "volume_ma"])
