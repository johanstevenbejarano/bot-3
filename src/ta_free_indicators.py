"""Reimplementaciones de EMA/RSI/ADX/ATR/Bollinger sin depender de la librería `ta`.

`ta==0.11.0` tiene un `setup.py` heredado que falla al compilar en algunos entornos limpios
(`AttributeError: install_layout` con setuptools/distutils modernos) — se confirmó en la
primera corrida real del agente programado en la nube. En vez de pelear con la instalación de
una dependencia frágil que además es la única no-mainstream del pipeline en vivo, se reimplementan
acá las fórmulas estándar (mismo suavizado de Wilder que usa `ta` para RSI/ATR/ADX) directo con
pandas/numpy — sin ninguna dependencia adicional.

Los módulos de backtesting (`indicators.py`, `meanrev_indicators.py`, etc.) siguen usando `ta`
sin cambios — no hay que arriesgar alterar ningún resultado ya documentado en FINDINGS.md. Esto
se usa SOLO en `live_snapshot.py`.
"""
from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0)
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0)

    atr_val = atr(high, low, close, window)
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr_val
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr_val

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Devuelve (basis, upper, lower)."""
    basis = close.rolling(window).mean()
    std = close.rolling(window).std()
    return basis, basis + num_std * std, basis - num_std * std
