"""Análisis de MAE/MFE (Maximum Adverse/Favorable Excursion) sobre las señales de entrada.

Para cada barra donde la estrategia dispararía una entrada, mide cuánto se mueve el precio
en contra (MAE) y a favor (MFE) durante una ventana de tenencia fija, en unidades de ATR al
momento de la entrada. Sirve para calibrar el SL/TP con datos reales de volatilidad en vez de
adivinar multiplicadores de ATR — un SL más ajustado que el MAE típico solo genera stop-outs
por ruido, no por señales realmente invalidadas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_mae_mfe(df_with_signal: pd.DataFrame, holding_bars: int = 72) -> pd.DataFrame:
    """Devuelve un DataFrame con mae_atr y mfe_atr (en múltiplos de ATR) por señal disparada."""
    signal_idx = np.flatnonzero(df_with_signal["signal"].to_numpy())
    close = df_with_signal["close"].to_numpy()
    high = df_with_signal["high"].to_numpy()
    low = df_with_signal["low"].to_numpy()
    atr = df_with_signal["atr"].to_numpy()
    n = len(df_with_signal)

    rows = []
    for i in signal_idx:
        end = min(i + 1 + holding_bars, n)
        if end <= i + 1:
            continue
        entry = close[i]
        entry_atr = atr[i]
        if entry_atr <= 0:
            continue
        window_low = low[i + 1:end].min()
        window_high = high[i + 1:end].max()
        mae_atr = (window_low - entry) / entry_atr
        mfe_atr = (window_high - entry) / entry_atr
        rows.append({"timestamp": df_with_signal.index[i], "mae_atr": mae_atr, "mfe_atr": mfe_atr})

    return pd.DataFrame(rows)


def summarize_excursions(mae_mfe_df: pd.DataFrame) -> dict:
    if mae_mfe_df.empty:
        return {"n_signals": 0}
    percentiles = [10, 25, 50, 75, 90]
    return {
        "n_signals": len(mae_mfe_df),
        "mae_atr_percentiles": {
            f"p{p}": float(np.percentile(mae_mfe_df["mae_atr"], p)) for p in percentiles
        },
        "mfe_atr_percentiles": {
            f"p{p}": float(np.percentile(mae_mfe_df["mfe_atr"], p)) for p in percentiles
        },
    }
