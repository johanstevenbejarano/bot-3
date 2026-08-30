"""Filtro de tendencia de timeframe mayor (higher-timeframe): exige que la señal de un timeframe
rápido (1h) esté a favor de la tendencia de uno más lento (por defecto diario) antes de operar.

Motivación: el análisis MAE/MFE de la línea 1 (ver FINDINGS.md) mostró que en BTC/ETH a 1h la
excursión adversa mediana antes de que el precio siga a favor es de ~3.5-4x ATR — mucho ruido de
corto plazo. Alinear con la tendencia de un timeframe mayor es una forma barata y bien establecida
de filtrar señales que van contra el movimiento de fondo, sin inventar una fuente de datos nueva.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import ta

from src.data_fetch import fetch_ohlcv


@dataclass(frozen=True)
class HTFTrendConfig:
    timeframe: str = "1d"
    ema_period: int = 50


def _shift_and_align(daily_bool_df: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Desplaza 1 vela (la del timeframe mayor ya cerrada) y la propaga sobre `target_index`.

    Sin este shift, una barra de 1h dentro del día D vería la tendencia calculada con el cierre
    del propio día D — que todavía no existe en ese momento. Con el shift, cualquier hora del
    día D usa la tendencia confirmada al cierre del día D-1, que es lo único que se sabía en
    tiempo real a esa hora.
    """
    return daily_bool_df.shift(1).reindex(target_index, method="ffill")


def compute_htf_trend(htf_close: pd.Series, ema_period: int) -> pd.DataFrame:
    """Booleans trend_up_htf / trend_dn_htf sobre el propio índice del timeframe mayor."""
    ema = ta.trend.EMAIndicator(htf_close, window=ema_period).ema_indicator()
    return pd.DataFrame(
        {"trend_up_htf": htf_close > ema, "trend_dn_htf": htf_close < ema}
    ).dropna()


def add_htf_trend(
    df_1h: pd.DataFrame, symbol: str, cfg: HTFTrendConfig, years: int
) -> pd.DataFrame:
    """Añade trend_up_htf / trend_dn_htf a `df_1h`, usando solo la vela del timeframe mayor ya
    confirmada al momento de cada barra de 1h (ver `_shift_and_align`) — sin lookahead.
    """
    htf = fetch_ohlcv(symbol, timeframe=cfg.timeframe, years=years)
    htf_trend = compute_htf_trend(htf["close"], cfg.ema_period)
    aligned = _shift_and_align(htf_trend, df_1h.index)

    out = df_1h.copy()
    out["trend_up_htf"] = aligned["trend_up_htf"]
    out["trend_dn_htf"] = aligned["trend_dn_htf"]
    return out.dropna(subset=["trend_up_htf", "trend_dn_htf"])
