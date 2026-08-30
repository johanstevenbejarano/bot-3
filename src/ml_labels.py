"""Labeling triple-barrera: para cada barra, simula hacia adelante un long y un short
hipotéticos con SL/TP vía ATR (misma mecánica que todas las estrategias de la sesión) y registra
cuál barrera se toca primero — la etiqueta que usa el clasificador para aprender, y también el
PnL real usado para simular la estrategia sin volver a correr un backtest completo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    df: pd.DataFrame,
    sl_atr_mult: float,
    tp_atr_mult: float,
    max_holding_bars: int,
    commission: float = 0.0,
) -> pd.DataFrame:
    """`commission` es la comisión de un solo fill (ej. 0.001 = 0.1%, la de Binance spot/taker).
    Se descuenta 2x (entrada + salida) de cada PnL — sin esto, el modelo aprendería una noción
    de "trade ganador" que no existe en la práctica una vez pagados los costos reales.
    """
    round_trip_cost_pct = 2 * commission * 100
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr = df["atr"].to_numpy()
    n = len(df)

    long_pnl = np.full(n, np.nan)
    long_offset = np.full(n, -1, dtype=int)
    short_pnl = np.full(n, np.nan)
    short_offset = np.full(n, -1, dtype=int)

    # t es válido si t + max_holding_bars <= n - 1 (hay suficiente futuro para resolver ambas
    # barreras o llegar a la barrera de tiempo) -> t < n - max_holding_bars.
    last_t = max(n - max_holding_bars, 0)
    for t in range(last_t):
        entry = close[t]
        a = atr[t]
        if a <= 0:
            continue

        tp_l, sl_l = entry + tp_atr_mult * a, entry - sl_atr_mult * a
        tp_s, sl_s = entry - tp_atr_mult * a, entry + sl_atr_mult * a
        long_done = short_done = False

        for h in range(1, max_holding_bars + 1):
            i = t + h
            hi, lo = high[i], low[i]

            if not long_done:
                hit_sl, hit_tp = lo <= sl_l, hi >= tp_l
                if hit_sl or hit_tp:
                    # si ambas barreras se tocan en la misma vela, asume SL primero (conservador)
                    long_pnl[t] = ((sl_l if hit_sl else tp_l) / entry - 1) * 100
                    long_offset[t] = h
                    long_done = True

            if not short_done:
                hit_sl, hit_tp = hi >= sl_s, lo <= tp_s
                if hit_sl or hit_tp:
                    exit_price = sl_s if hit_sl else tp_s
                    short_pnl[t] = (entry - exit_price) / entry * 100
                    short_offset[t] = h
                    short_done = True

            if long_done and short_done:
                break

        if not long_done:
            long_pnl[t] = (close[t + max_holding_bars] / entry - 1) * 100
            long_offset[t] = max_holding_bars
        if not short_done:
            short_pnl[t] = (entry - close[t + max_holding_bars]) / entry * 100
            short_offset[t] = max_holding_bars

    out = df.copy()
    out["long_pnl_pct"] = long_pnl - round_trip_cost_pct
    out["long_exit_offset"] = long_offset
    out["short_pnl_pct"] = short_pnl - round_trip_cost_pct
    out["short_exit_offset"] = short_offset
    return out.iloc[:last_t]
