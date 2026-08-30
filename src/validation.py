"""Separación train/test por tiempo y búsqueda de parámetros validada fuera de muestra.

Objetivo: evitar el sobreajuste que aparece al calibrar SL/TP y thresholds de entrada
directamente sobre todo el histórico. La búsqueda de parámetros solo puede ver `train`;
`test` se evalúa una única vez, con la configuración ya elegida.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import NamedTuple

import pandas as pd
from backtesting.lib import FractionalBacktest

from src.config import BACKTEST, STRATEGY, StrategyConfig
from src.strategy import ConfluenceStrategy, compute_layers


def train_test_split_by_time(
    df: pd.DataFrame, train_fraction: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide una serie temporal ya ordenada en train/test por fecha de corte (sin barajar)."""
    cutoff_idx = int(len(df) * train_fraction)
    return df.iloc[:cutoff_idx], df.iloc[cutoff_idx:]


class RunResult(NamedTuple):
    num_trades: int
    win_rate_pct: float
    expectancy_pct: float
    return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float


def _to_backtest_frame(df_with_layers: pd.DataFrame) -> pd.DataFrame:
    bt_data = df_with_layers.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    bt_data["atr"] = bt_data["atr"] * BACKTEST.fractional_unit
    bt_data["swing_low"] = bt_data["swing_low"] * BACKTEST.fractional_unit
    bt_data["swing_high"] = bt_data["swing_high"] * BACKTEST.fractional_unit
    return bt_data


def run_config(
    df_with_indicators: pd.DataFrame,
    cfg: StrategyConfig,
    sl_anchor: str = "entry",
) -> RunResult:
    """Corre un backtest con la config de estrategia dada sobre un tramo (train o test)."""
    with_signal = compute_layers(df_with_indicators, cfg)
    bt_data = _to_backtest_frame(with_signal)

    bt = FractionalBacktest(
        bt_data,
        ConfluenceStrategy,
        cash=BACKTEST.initial_cash,
        commission=BACKTEST.commission,
        exclusive_orders=True,
        fractional_unit=BACKTEST.fractional_unit,
    )
    stats = bt.run(
        sl_anchor=sl_anchor,
        sl_atr_mult=cfg.risk.sl_atr_mult,
        tp_atr_mult=cfg.risk.tp_atr_mult,
    )
    trades = stats["_trades"]
    n = len(trades)
    return RunResult(
        num_trades=n,
        win_rate_pct=float(stats["Win Rate [%]"]) if n else 0.0,
        expectancy_pct=float(trades["ReturnPct"].mean() * 100) if n else 0.0,
        return_pct=float(stats["Return [%]"]),
        max_drawdown_pct=float(stats["Max. Drawdown [%]"]),
        sharpe_ratio=float(stats["Sharpe Ratio"]) if n else float("nan"),
    )


def grid_search_best_config(
    train_by_symbol: dict[str, pd.DataFrame],
    adx_values: tuple[float, ...] = (15, 20, 25),
    lookback_values: tuple[int, ...] = (2, 3, 5),
    # Rango informado por el análisis de MAE/MFE (src/analyze_excursions.py): la excursión
    # adversa mediana antes de que el precio siga a favor es de ~3.5-4x ATR, muy por encima
    # del 1.5x usado originalmente — de ahí que el rango de SL parta más ancho.
    sl_atr_mults: tuple[float, ...] = (3.0, 3.5, 4.0, 4.5),
    tp_atr_mults: tuple[float, ...] = (4.0, 5.0, 6.0, 7.0, 8.0),
    min_trades_per_symbol: int = 15,
    sl_anchor: str = "entry",
) -> tuple[StrategyConfig | None, dict]:
    """Busca, SOLO en `train`, la config con mejor expectancy mínima consistente entre símbolos.

    Requiere expectancy > 0 en TODOS los símbolos (no promedio) para evitar elegir una config
    que solo funciona en un par por azar. Devuelve None si ninguna combinación lo logra.
    """
    best_cfg = None
    best_score = float("-inf")
    best_detail: dict = {}

    for adx_th, lookback, sl_mult, tp_mult in product(
        adx_values, lookback_values, sl_atr_mults, tp_atr_mults
    ):
        cfg = replace(
            STRATEGY,
            trend=replace(STRATEGY.trend, adx_threshold=adx_th),
            pullback=replace(STRATEGY.pullback, lookback=lookback),
            risk=replace(STRATEGY.risk, sl_atr_mult=sl_mult, tp_atr_mult=tp_mult),
        )

        per_symbol = {}
        for symbol, df in train_by_symbol.items():
            per_symbol[symbol] = run_config(df, cfg, sl_anchor=sl_anchor)

        if any(r.num_trades < min_trades_per_symbol for r in per_symbol.values()):
            continue
        expectancies = [r.expectancy_pct for r in per_symbol.values()]
        if min(expectancies) <= 0:
            continue

        score = min(expectancies)  # el peor caso entre símbolos, no el promedio
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_detail = {s: r._asdict() for s, r in per_symbol.items()}

    return best_cfg, best_detail


def grid_search_best_config_per_symbol(
    train_by_symbol: dict[str, pd.DataFrame],
    adx_values: tuple[float, ...] = (15, 20, 25, 30),
    lookback_values: tuple[int, ...] = (2, 3, 5),
    sl_atr_mults: tuple[float, ...] = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5),
    tp_atr_mults: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0),
    min_trades: int = 15,
    sl_anchor: str = "entry",
) -> dict[str, tuple[StrategyConfig | None, RunResult | None]]:
    """Busca, SOLO en `train`, la mejor config para CADA símbolo por separado.

    A diferencia de `grid_search_best_config`, no exige que la misma config funcione en ambos
    pares — BTC y ETH no son activos idénticos pese a su alta correlación. El score usado es
    expectancy_pct (a mayor, mejor), con un piso de trades mínimo para evitar elegir por ruido
    de muestra pequeña.
    """
    results: dict[str, tuple[StrategyConfig | None, RunResult | None]] = {}

    for symbol, df in train_by_symbol.items():
        best_cfg = None
        best_result: RunResult | None = None
        best_score = float("-inf")

        for adx_th, lookback, sl_mult, tp_mult in product(
            adx_values, lookback_values, sl_atr_mults, tp_atr_mults
        ):
            cfg = replace(
                STRATEGY,
                trend=replace(STRATEGY.trend, adx_threshold=adx_th),
                pullback=replace(STRATEGY.pullback, lookback=lookback),
                risk=replace(STRATEGY.risk, sl_atr_mult=sl_mult, tp_atr_mult=tp_mult),
            )
            result = run_config(df, cfg, sl_anchor=sl_anchor)

            if result.num_trades < min_trades or result.expectancy_pct <= 0:
                continue
            if result.expectancy_pct > best_score:
                best_score = result.expectancy_pct
                best_cfg = cfg
                best_result = result

        results[symbol] = (best_cfg, best_result)

    return results
