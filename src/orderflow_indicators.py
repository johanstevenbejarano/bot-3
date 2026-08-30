"""Indicadores para la estrategia de flujo institucional: percentil rodante del interés abierto
(Bybit) + z-score rodante del sesgo de flujo de órdenes (taker buy ratio, Binance), alineados
sobre el índice de 1h de precio/volumen (Binance) — mismo criterio de alineación que
`funding_indicators.py` (ffill del último valor conocido, sin mirar al futuro).
"""
from __future__ import annotations

import pandas as pd
import ta

from src.config import OrderflowConfig


def compute_oi_percentile(open_interest: pd.Series, lookback_periods: int) -> pd.Series:
    """Percentil rodante del interés abierto dentro de su propia ventana histórica reciente."""
    return open_interest.rolling(window=lookback_periods, min_periods=lookback_periods // 2).rank(pct=True)


def compute_imbalance_zscore(taker_buy_ratio: pd.Series, lookback_periods: int) -> pd.Series:
    """Cuántos desvíos estándar se aleja el ratio de compra/venta agresiva de su propia media
    reciente -- un z-score alto = sesgo comprador inusual, uno muy negativo = sesgo vendedor."""
    rolling_mean = taker_buy_ratio.rolling(window=lookback_periods, min_periods=lookback_periods // 2).mean()
    rolling_std = taker_buy_ratio.rolling(window=lookback_periods, min_periods=lookback_periods // 2).std()
    return (taker_buy_ratio - rolling_mean) / rolling_std.replace(0, pd.NA)


def align_orderflow_to_1h(
    df_1h: pd.DataFrame, oi_percentile: pd.Series, imbalance_zscore: pd.Series
) -> pd.DataFrame:
    """Propaga ambas series (posiblemente con huecos u origen en otro exchange) hacia adelante
    sobre el índice de 1h de precio -- el último valor conocido es lo único disponible en cada
    barra, igual que `funding_indicators.align_funding_to_1h`."""
    out = df_1h.copy()
    out["oi_percentile"] = oi_percentile.reindex(df_1h.index, method="ffill")
    out["imbalance_zscore"] = imbalance_zscore.reindex(df_1h.index, method="ffill")
    return out


def add_orderflow_risk_indicators(df: pd.DataFrame, cfg: OrderflowConfig) -> pd.DataFrame:
    """Añade ATR y media de volumen -- mismas columnas que el resto de las líneas de estrategia."""
    out = df.copy()
    out["atr"] = ta.volatility.AverageTrueRange(
        out["high"], out["low"], out["close"], window=cfg.atr_period
    ).average_true_range()
    out["volume_ma"] = out["volume"].rolling(window=cfg.volume.volume_ma_period).mean()
    return out


def add_orderflow_indicators(df_1h: pd.DataFrame, symbol: str, cfg: OrderflowConfig, years: int) -> pd.DataFrame:
    """Pipeline completo: trae open interest (Bybit) + taker buy ratio (Binance), calcula
    percentil/z-score, alinea sobre el índice de 1h del precio, y añade ATR/volumen."""
    from src.orderflow_data import fetch_open_interest_bybit, fetch_taker_buy_ratio_binance

    open_interest = fetch_open_interest_bybit(symbol, years=years)
    taker_buy_ratio = fetch_taker_buy_ratio_binance(symbol, years=years)

    oi_percentile = compute_oi_percentile(open_interest, cfg.oi_lookback_periods)
    imbalance_zscore = compute_imbalance_zscore(taker_buy_ratio, cfg.imbalance_lookback_periods)

    out = align_orderflow_to_1h(df_1h, oi_percentile, imbalance_zscore)
    out = add_orderflow_risk_indicators(out, cfg)
    return out.dropna(subset=["oi_percentile", "imbalance_zscore", "atr", "volume_ma"])
