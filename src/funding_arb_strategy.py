"""Arbitraje de funding rate (cash-and-carry): comprar spot + vender el perpetuo al mismo tiempo
(posición neutral al precio) cuando el funding rate es lo bastante positivo como para que el pago
que se cobra (los largos le pagan a los cortos) supere el costo de operar. A diferencia de las 9
líneas anteriores, esto NO es una apuesta direccional -- el riesgo de precio de la pata larga
(spot) y la pata corta (perpetuo) se cancelan, salvo por el cambio en la "base" (diferencia
spot-perpetuo) entre la entrada y la salida, que es el riesgo real de esta estrategia.
"""
from __future__ import annotations

import pandas as pd


def find_episodes(
    funding_rate: pd.Series, entry_threshold: float, exit_threshold: float
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Identifica ventanas continuas [entrada, salida) donde la posición estaría abierta: entra
    cuando el funding supera `entry_threshold`, sale cuando cae a `exit_threshold` o menos
    (histéresis simple para no entrar/salir por ruido alrededor de un solo umbral). Un tramo que
    sigue abierto al final de la serie se descarta (no hay precio de salida real que usar)."""
    episodes: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_position = False
    entry_time: pd.Timestamp | None = None

    for timestamp, rate in funding_rate.items():
        if not in_position and rate > entry_threshold:
            in_position = True
            entry_time = timestamp
        elif in_position and rate <= exit_threshold:
            episodes.append((entry_time, timestamp))
            in_position = False
            entry_time = None

    return episodes


def compute_episode_return_pct(
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    spot: pd.Series,
    perp: pd.Series,
    funding_rate: pd.Series,
    commission: float,
) -> float:
    """Retorno % de un episodio: cambio en la base (riesgo de precio real de esta estrategia) +
    funding cobrado durante la ventana - comisión de las 4 operaciones (abrir spot+perp, cerrar
    spot+perp)."""
    entry_spot = float(spot.asof(entry_time))
    entry_perp = float(perp.asof(entry_time))
    exit_spot = float(spot.asof(exit_time))
    exit_perp = float(perp.asof(exit_time))

    entry_basis = entry_perp - entry_spot
    exit_basis = exit_perp - exit_spot
    price_pnl_pct = (entry_basis - exit_basis) / entry_spot * 100

    funding_window = funding_rate[(funding_rate.index >= entry_time) & (funding_rate.index < exit_time)]
    funding_pnl_pct = float(funding_window.sum()) * 100

    cost_pct = commission * 4 * 100

    return price_pnl_pct + funding_pnl_pct - cost_pct


def backtest_funding_arb(
    spot: pd.Series,
    perp: pd.Series,
    funding_rate: pd.Series,
    entry_threshold: float,
    exit_threshold: float,
    commission: float,
) -> list[float]:
    """Retorno % de cada episodio de la estrategia sobre todo el período disponible. Descarta
    episodios cuya entrada/salida cae antes del inicio real de `spot`/`perp` (ej. el perpetuo
    suele tener menos historia que el spot) -- `.asof()` en ese caso da NaN, y un solo NaN
    arruinaría el promedio de toda la serie si no se filtra."""
    episodes = find_episodes(funding_rate, entry_threshold, exit_threshold)
    returns = [
        compute_episode_return_pct(entry, exit_, spot, perp, funding_rate, commission)
        for entry, exit_ in episodes
    ]
    return [r for r in returns if pd.notna(r)]
