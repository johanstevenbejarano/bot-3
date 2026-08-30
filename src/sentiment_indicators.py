"""Alinea el Fear & Greed Index (evento diario) sobre velas de 1h, y calcula ATR/volumen para la
estrategia contraria de sentimiento extremo -- mismo criterio de alineación que
`funding_indicators.py` (ffill del último valor conocido, sin mirar al futuro).
"""
from __future__ import annotations

import pandas as pd
import ta

from src.config import SentimentConfig


def align_sentiment_to_1h(df_1h: pd.DataFrame, fear_greed: pd.Series) -> pd.DataFrame:
    aligned = fear_greed.reindex(df_1h.index, method="ffill")
    out = df_1h.copy()
    out["fear_greed"] = aligned
    return out


def add_sentiment_indicators(df_1h: pd.DataFrame, cfg: SentimentConfig) -> pd.DataFrame:
    """Pipeline completo: trae el Fear & Greed Index, lo alinea sobre el índice de 1h del precio,
    y añade ATR/volumen -- mismas columnas que el resto de las líneas de estrategia."""
    from src.sentiment_data import fetch_fear_greed_index

    fear_greed = fetch_fear_greed_index()
    out = align_sentiment_to_1h(df_1h, fear_greed)

    out["atr"] = ta.volatility.AverageTrueRange(
        out["high"], out["low"], out["close"], window=cfg.atr_period
    ).average_true_range()
    out["volume_ma"] = out["volume"].rolling(window=cfg.volume.volume_ma_period).mean()

    return out.dropna(subset=["fear_greed", "atr", "volume_ma"])
