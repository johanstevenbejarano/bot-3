"""Percentil rodante del DXY (evento diario, mercado bursátil cierra fines de semana/feriados) y
su alineación sobre velas de 1h (cripto opera 24/7) -- mismo criterio de alineación que
`funding_indicators.py` (ffill del último valor conocido, incluyendo durante fines de semana en
que el dólar no cotiza pero cripto sí sigue operando).
"""
from __future__ import annotations

import pandas as pd
import ta

from src.config import MacroConfig


def compute_dxy_percentile(dxy: pd.Series, lookback_periods: int) -> pd.Series:
    """Percentil rodante del DXY dentro de su propia ventana histórica reciente (en días, ya que
    el DXY es una serie diaria)."""
    return dxy.rolling(window=lookback_periods, min_periods=lookback_periods // 2).rank(pct=True)


def align_dxy_to_1h(df_1h: pd.DataFrame, dxy_percentile: pd.Series) -> pd.DataFrame:
    aligned = dxy_percentile.reindex(df_1h.index, method="ffill")
    out = df_1h.copy()
    out["dxy_percentile"] = aligned
    return out


def add_macro_indicators(df_1h: pd.DataFrame, cfg: MacroConfig) -> pd.DataFrame:
    """Pipeline completo: trae el DXY, calcula su percentil rodante, lo alinea sobre el índice de
    1h del precio, y añade ATR/volumen -- mismas columnas que el resto de las líneas."""
    from src.macro_data import fetch_dxy_daily

    dxy = fetch_dxy_daily()
    dxy_percentile = compute_dxy_percentile(dxy, cfg.lookback_periods)
    out = align_dxy_to_1h(df_1h, dxy_percentile)

    out["atr"] = ta.volatility.AverageTrueRange(
        out["high"], out["low"], out["close"], window=cfg.atr_period
    ).average_true_range()
    out["volume_ma"] = out["volume"].rolling(window=cfg.volume.volume_ma_period).mean()

    return out.dropna(subset=["dxy_percentile", "atr", "volume_ma"])
