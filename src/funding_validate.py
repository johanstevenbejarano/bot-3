"""Busca parámetros de la estrategia contraria de funding SOLO en train y valida en test — misma
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

from src.config import BACKTEST, FUNDING_STRATEGY, RESULTS_DIR, FundingConfig
from src.data_fetch import fetch_ohlcv
from src.funding_indicators import add_funding_indicators
from src.funding_strategy import FundingContrarianStrategy, compute_layers
from src.validation import train_test_split_by_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SYMBOL = "BTC/USDT"
YEARS_OF_HISTORY = 5
TRAIN_FRACTION = 0.7


class RunResult(NamedTuple):
    num_trades: int
    win_rate_pct: float
    expectancy_pct: float
    return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float


def run_config(df_with_indicators: pd.DataFrame, cfg: FundingConfig) -> RunResult:
    with_signal = compute_layers(df_with_indicators, cfg)
    bt_data = with_signal.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    bt_data["atr"] = bt_data["atr"] * BACKTEST.fractional_unit

    bt = FractionalBacktest(
        bt_data, FundingContrarianStrategy,
        cash=BACKTEST.initial_cash, commission=BACKTEST.commission,
        exclusive_orders=True, fractional_unit=BACKTEST.fractional_unit,
    )
    stats = bt.run(sl_atr_mult=cfg.sl_atr_mult, tp_atr_mult=cfg.tp_atr_mult)
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


def run_on_raw(raw_1h: pd.DataFrame, cfg: FundingConfig) -> RunResult:
    with_indicators = add_funding_indicators(raw_1h, SYMBOL, cfg, years=YEARS_OF_HISTORY)
    return run_config(with_indicators, cfg)


def grid_search_single_symbol(
    raw_with_funding: pd.DataFrame,
    lookback_values: tuple[int, ...] = (90, 180, 270, 540),
    extreme_percentiles: tuple[float, ...] = (0.85, 0.90, 0.95),
    sl_atr_mults: tuple[float, ...] = (2.0, 3.0, 4.0),
    tp_atr_mults: tuple[float, ...] = (4.0, 6.0, 8.0),
    min_trades: int = 20,
) -> FundingConfig | None:
    """Busca la mejor config sobre un DataFrame que YA tiene `funding_percentile` calculado con
    el lookback base — reescalar `lookback_periods` en el grid recalcularía el percentil desde
    cero por combinación (más lento); para mantenerlo simple, el lookback se fija por corrida.
    """
    best_cfg = None
    best_score = float("-inf")

    for extreme_pct, sl_mult, tp_mult in product(extreme_percentiles, sl_atr_mults, tp_atr_mults):
        cfg = replace(
            FUNDING_STRATEGY,
            extreme_percentile=extreme_pct, sl_atr_mult=sl_mult, tp_atr_mult=tp_mult,
        )
        with_signal = compute_layers(raw_with_funding, cfg)
        bt_data = with_signal.rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        )
        bt_data["atr"] = bt_data["atr"] * BACKTEST.fractional_unit
        bt = FractionalBacktest(
            bt_data, FundingContrarianStrategy,
            cash=BACKTEST.initial_cash, commission=BACKTEST.commission,
            exclusive_orders=True, fractional_unit=BACKTEST.fractional_unit,
        )
        stats = bt.run(sl_atr_mult=sl_mult, tp_atr_mult=tp_mult)
        trades = stats["_trades"]
        n = len(trades)
        if n < min_trades:
            continue
        expectancy = float(trades["ReturnPct"].mean() * 100)
        if expectancy <= 0:
            continue
        if expectancy > best_score:
            best_score = expectancy
            best_cfg = cfg

    return best_cfg


def main() -> None:
    raw = fetch_ohlcv(SYMBOL, timeframe=BACKTEST.timeframe, years=YEARS_OF_HISTORY)
    with_funding = add_funding_indicators(raw, SYMBOL, FUNDING_STRATEGY, years=YEARS_OF_HISTORY)

    train_df, test_df = train_test_split_by_time(with_funding, TRAIN_FRACTION)
    logger.info(
        "train: %s -> %s (%d velas), test: %s -> %s (%d velas)",
        train_df.index[0], train_df.index[-1], len(train_df),
        test_df.index[0], test_df.index[-1], len(test_df),
    )

    logger.info("Buscando config en TRAIN unicamente (test no se toca todavia)...")
    best_cfg = grid_search_single_symbol(train_df)

    if best_cfg is None:
        report = {"status": "no_viable_config", "message": "Ninguna combinacion da expectancy positiva en train."}
        logger.warning(report["message"])
    else:
        train_result = run_config(train_df, best_cfg)
        test_result = run_config(test_df, best_cfg)
        holds_up = test_result.expectancy_pct > 0

        logger.info("Mejor config en TRAIN: %s", asdict(best_cfg))
        logger.info("TRAIN: %s", train_result._asdict())
        logger.info("TEST: %s -> %s", test_result._asdict(), "SOSTIENE" if holds_up else "NO sostiene")

        report = {
            "status": "validated" if holds_up else "failed_out_of_sample",
            "config": asdict(best_cfg),
            "train_metrics": train_result._asdict(),
            "test_metrics": test_result._asdict(),
            "holds_up_out_of_sample": holds_up,
        }

    out_path = RESULTS_DIR / "funding_train_test_validation.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte guardado en %s", out_path)


if __name__ == "__main__":
    main()
