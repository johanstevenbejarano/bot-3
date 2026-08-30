"""Busca parámetros de la estrategia de reversión a la media SOLO en train y valida en test —
misma disciplina que las líneas anteriores (ver FINDINGS.md).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from itertools import product
from typing import NamedTuple

import pandas as pd
from backtesting.lib import FractionalBacktest

from src.config import BACKTEST, MEANREV_STRATEGY, PAIRS, RESULTS_DIR, MeanReversionConfig
from src.data_fetch import fetch_ohlcv
from src.meanrev_indicators import add_meanrev_indicators
from src.meanrev_strategy import MeanReversionStrategy, compute_layers
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


def run_config(df_with_indicators: pd.DataFrame, cfg: MeanReversionConfig) -> RunResult:
    with_signal = compute_layers(df_with_indicators, cfg)
    bt_data = with_signal.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    bt_data["atr"] = bt_data["atr"] * BACKTEST.fractional_unit
    bt_data["bb_basis"] = bt_data["bb_basis"] * BACKTEST.fractional_unit

    bt = FractionalBacktest(
        bt_data, MeanReversionStrategy,
        cash=BACKTEST.initial_cash, commission=BACKTEST.commission,
        exclusive_orders=True, fractional_unit=BACKTEST.fractional_unit,
    )
    stats = bt.run(sl_atr_mult=cfg.risk.sl_atr_mult)
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


def _train_slice(raw: pd.DataFrame, cfg: MeanReversionConfig) -> pd.DataFrame:
    with_indicators = add_meanrev_indicators(raw, cfg)
    train_df, _ = train_test_split_by_time(with_indicators, TRAIN_FRACTION)
    return train_df


def _test_slice(raw: pd.DataFrame, cfg: MeanReversionConfig) -> pd.DataFrame:
    with_indicators = add_meanrev_indicators(raw, cfg)
    _, test_df = train_test_split_by_time(with_indicators, TRAIN_FRACTION)
    return test_df


def grid_search_best_config(
    raw_by_symbol: dict[str, pd.DataFrame],
    num_std_values: tuple[float, ...] = (2.0, 2.5, 3.0),
    rsi_extreme_values: tuple[tuple[float, float], ...] = ((30, 70), (25, 75), (20, 80)),
    sl_atr_mults: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0),
    min_trades_per_symbol: int = 30,
) -> tuple[MeanReversionConfig | None, dict]:
    best_cfg = None
    best_score = float("-inf")
    best_detail: dict = {}

    for num_std, (oversold, overbought), sl_mult in product(
        num_std_values, rsi_extreme_values, sl_atr_mults
    ):
        cfg = replace(
            MEANREV_STRATEGY,
            bollinger=replace(MEANREV_STRATEGY.bollinger, num_std=num_std),
            rsi=replace(MEANREV_STRATEGY.rsi, oversold=oversold, overbought=overbought),
            risk=replace(MEANREV_STRATEGY.risk, sl_atr_mult=sl_mult),
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

    out_path = RESULTS_DIR / "meanrev_train_test_validation.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte guardado en %s", out_path)


if __name__ == "__main__":
    main()
