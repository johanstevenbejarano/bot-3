"""Busca parámetros de la estrategia de breakout SOLO en train y valida en test — misma
disciplina que las líneas anteriores (ver FINDINGS.md).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from itertools import product
from typing import NamedTuple

import pandas as pd
from backtesting.lib import FractionalBacktest

from src.breakout_indicators import add_breakout_indicators
from src.breakout_strategy import BreakoutStrategy, compute_layers
from src.config import BACKTEST, BREAKOUT_STRATEGY, PAIRS, RESULTS_DIR, BreakoutConfig
from src.data_fetch import fetch_ohlcv
from src.validation import train_test_split_by_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TRAIN_FRACTION = 0.7


class RunResult(NamedTuple):
    num_trades: int
    win_rate_pct: float
    expectancy_pct: float
    return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float


def run_config(df_with_indicators: pd.DataFrame, cfg: BreakoutConfig) -> RunResult:
    with_signal = compute_layers(df_with_indicators, cfg)
    bt_data = with_signal.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    bt_data["atr"] = bt_data["atr"] * BACKTEST.fractional_unit

    bt = FractionalBacktest(
        bt_data, BreakoutStrategy,
        cash=BACKTEST.initial_cash, commission=BACKTEST.commission,
        exclusive_orders=True, fractional_unit=BACKTEST.fractional_unit,
    )
    stats = bt.run(sl_atr_mult=cfg.risk.sl_atr_mult, tp_atr_mult=cfg.risk.tp_atr_mult)
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


def _train_slice(raw: pd.DataFrame, cfg: BreakoutConfig) -> pd.DataFrame:
    with_indicators = add_breakout_indicators(raw, cfg)
    train_df, _ = train_test_split_by_time(with_indicators, TRAIN_FRACTION)
    return train_df


def _test_slice(raw: pd.DataFrame, cfg: BreakoutConfig) -> pd.DataFrame:
    with_indicators = add_breakout_indicators(raw, cfg)
    _, test_df = train_test_split_by_time(with_indicators, TRAIN_FRACTION)
    return test_df


def run_on_raw(raw_slice: pd.DataFrame, cfg: BreakoutConfig) -> RunResult:
    """Corre la estrategia sobre un tramo de datos crudos arbitrario, sin hacer ningún split
    train/test propio — usado por walk_forward.py, que controla las ventanas desde afuera.
    """
    with_indicators = add_breakout_indicators(raw_slice, cfg)
    return run_config(with_indicators, cfg)


def grid_search_single_symbol(
    raw: pd.DataFrame,
    donchian_periods: tuple[int, ...] = (10, 20, 40, 55),
    sl_atr_mults: tuple[float, ...] = (2.0, 3.0, 4.0),
    tp_atr_mults: tuple[float, ...] = (4.0, 6.0, 8.0, 10.0),
    min_trades: int = 20,
) -> BreakoutConfig | None:
    """Busca la mejor config para UN símbolo sobre un tramo de datos crudos arbitrario (sin
    split interno) — usado por walk_forward.py, una vez por cada ventana de entrenamiento.
    """
    best_cfg = None
    best_score = float("-inf")

    for period, sl_mult, tp_mult in product(donchian_periods, sl_atr_mults, tp_atr_mults):
        cfg = replace(
            BREAKOUT_STRATEGY,
            donchian=replace(BREAKOUT_STRATEGY.donchian, period=period),
            risk=replace(BREAKOUT_STRATEGY.risk, sl_atr_mult=sl_mult, tp_atr_mult=tp_mult),
        )
        result = run_on_raw(raw, cfg)
        if result.num_trades < min_trades or result.expectancy_pct <= 0:
            continue
        if result.expectancy_pct > best_score:
            best_score = result.expectancy_pct
            best_cfg = cfg

    return best_cfg


def grid_search_best_config(
    raw_by_symbol: dict[str, pd.DataFrame],
    donchian_periods: tuple[int, ...] = (10, 20, 40, 55),
    sl_atr_mults: tuple[float, ...] = (2.0, 3.0, 4.0),
    tp_atr_mults: tuple[float, ...] = (4.0, 6.0, 8.0, 10.0),
    min_trades_per_symbol: int = 30,
) -> tuple[BreakoutConfig | None, dict]:
    best_cfg = None
    best_score = float("-inf")
    best_detail: dict = {}

    for period, sl_mult, tp_mult in product(donchian_periods, sl_atr_mults, tp_atr_mults):
        cfg = replace(
            BREAKOUT_STRATEGY,
            donchian=replace(BREAKOUT_STRATEGY.donchian, period=period),
            risk=replace(BREAKOUT_STRATEGY.risk, sl_atr_mult=sl_mult, tp_atr_mult=tp_mult),
        )

        per_symbol = {
            symbol: run_config(_train_slice(raw, cfg), cfg) for symbol, raw in raw_by_symbol.items()
        }

        if any(r.num_trades < min_trades_per_symbol for r in per_symbol.values()):
            continue
        expectancies = [r.expectancy_pct for r in per_symbol.values()]
        if min(expectancies) <= 0:
            continue

        score = min(expectancies)
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_detail = {s: r._asdict() for s, r in per_symbol.items()}

    return best_cfg, best_detail


def main() -> None:
    raw_by_symbol = {
        symbol: fetch_ohlcv(symbol, timeframe=BACKTEST.timeframe, years=BACKTEST.years_of_history)
        for symbol in PAIRS
    }

    logger.info("Buscando config en TRAIN unicamente (test no se toca todavia)...")
    best_cfg, train_detail = grid_search_best_config(raw_by_symbol)

    if best_cfg is None:
        report = {
            "status": "no_viable_config",
            "message": "Ninguna combinacion probada da expectancy positiva en ambos pares en train.",
        }
        logger.warning(report["message"])
    else:
        logger.info("Mejor config en TRAIN: %s", asdict(best_cfg))
        logger.info("TRAIN detail: %s", json.dumps(train_detail, indent=2))

        test_detail = {
            symbol: run_config(_test_slice(raw, best_cfg), best_cfg)._asdict()
            for symbol, raw in raw_by_symbol.items()
        }
        logger.info("TEST detail: %s", json.dumps(test_detail, indent=2))

        holds_up = all(v["expectancy_pct"] > 0 for v in test_detail.values())
        report = {
            "status": "validated" if holds_up else "failed_out_of_sample",
            "config": asdict(best_cfg),
            "train_metrics": train_detail,
            "test_metrics": test_detail,
            "holds_up_out_of_sample": holds_up,
        }
        logger.info("%s", "SOSTIENE fuera de muestra" if holds_up else "NO sostiene fuera de muestra")

    out_path = RESULTS_DIR / "breakout_train_test_validation.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte guardado en %s", out_path)


if __name__ == "__main__":
    main()
