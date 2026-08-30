"""Análisis de estacionalidad (día de la semana / hora del día): ¿el retorno de la vela siguiente
tiene un sesgo direccional detectable según cuándo ocurre, más allá de lo que explicaría el azar?

A diferencia de las 5 líneas anteriores (todas del tipo "regla de entrada con SL/TP vía
backtesting.py"), esto es una prueba estadística directa sobre la hipótesis en sí -- comparar la
media de retornos de cada grupo contra cero, con el mismo bootstrap ya usado y testeado en
`ml_significance.py`. Si ningún grupo muestra nada, no tiene sentido envolver la idea en una
estrategia completa con SL/TP: se ahorra ese trabajo si el test barato ya la descarta.
"""
from __future__ import annotations

import pandas as pd

from src.ml_significance import bootstrap_mean_ci


def compute_forward_returns(close: pd.Series, horizon_bars: int = 1) -> pd.Series:
    """% de retorno desde el cierre de la vela actual hasta `horizon_bars` velas después --
    "si comprara al cierre de esta vela, qué retorno tendría" (no usa nada del futuro más allá
    de eso, es la variable que se está intentando explicar, no una feature de entrada).
    """
    return (close.shift(-horizon_bars) / close - 1) * 100


def analyze_by_group(
    forward_returns: pd.Series, group_key: pd.Series, min_samples: int = 30, n_boot: int = 5000
) -> dict[int, dict]:
    """Bootstrap CI de los retornos futuros, agrupados por `group_key` (hora del día, día de la
    semana, etc.) -- una entrada por cada valor del grupo con muestra suficiente."""
    df = pd.DataFrame({"forward_return": forward_returns, "group": group_key}).dropna()

    results: dict[int, dict] = {}
    for group_value, subset in df.groupby("group"):
        values = subset["forward_return"].to_numpy()
        if len(values) < min_samples:
            continue
        results[int(group_value)] = bootstrap_mean_ci(values, n_boot=n_boot)

    return results


def analyze_by_hour(df_1h: pd.DataFrame, horizon_bars: int = 1, min_samples: int = 30) -> dict[int, dict]:
    forward_returns = compute_forward_returns(df_1h["close"], horizon_bars)
    hour_of_day = pd.Series(df_1h.index.hour, index=df_1h.index)
    return analyze_by_group(forward_returns, hour_of_day, min_samples=min_samples)


def analyze_by_day_of_week(df_1h: pd.DataFrame, horizon_bars: int = 1, min_samples: int = 30) -> dict[int, dict]:
    forward_returns = compute_forward_returns(df_1h["close"], horizon_bars)
    day_of_week = pd.Series(df_1h.index.dayofweek, index=df_1h.index)  # 0=lunes ... 6=domingo
    return analyze_by_group(forward_returns, day_of_week, min_samples=min_samples)
