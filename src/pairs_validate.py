"""Busca los parámetros del pairs trade SOLO en train y valida la elegida en test — misma
disciplina train/test que el resto del proyecto (ver FINDINGS.md).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from itertools import product

import pandas as pd

from src.config import BACKTEST, PAIRS_STRATEGY, RESULTS_DIR
from src.data_fetch import fetch_ohlcv
from src.pairs_strategy import PairsConfig, backtest_pairs, compute_spread_zscore
from src.validation import train_test_split_by_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TRAIN_FRACTION = 0.7


def _run(df_a: pd.DataFrame, df_b: pd.DataFrame, cfg: PairsConfig):
    df = compute_spread_zscore(df_a, df_b, cfg)
    return backtest_pairs(df, cfg)


def grid_search_pairs(df_a_train, df_b_train, min_trades: int = 30):
    best_cfg = None
    best_stats = None
    best_score = float("-inf")

    for beta_w, z_w, entry_z, exit_z, stop_z in product(
        (480, 720, 1080, 2160),  # ~20, 30, 45, 90 días
        (240, 480, 720),  # ~10, 20, 30 días
        (1.5, 2.0, 2.5),
        (0.25, 0.5),
        (2.5, 3.0, 3.5),
    ):
        if exit_z >= entry_z or beta_w < z_w:
            continue
        cfg = replace(
            PAIRS_STRATEGY,
            beta_window=beta_w, z_window=z_w, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z,
        )
        _, stats = _run(df_a_train, df_b_train, cfg)

        if stats.num_trades < min_trades or stats.expectancy_pct <= 0:
            continue
        if stats.expectancy_pct > best_score:
            best_score = stats.expectancy_pct
            best_cfg = cfg
            best_stats = stats

    return best_cfg, best_stats


def main() -> None:
    df_a = fetch_ohlcv(
        PAIRS_STRATEGY.symbol_a, timeframe=BACKTEST.timeframe, years=PAIRS_STRATEGY.years_of_history
    )
    df_b = fetch_ohlcv(
        PAIRS_STRATEGY.symbol_b, timeframe=BACKTEST.timeframe, years=PAIRS_STRATEGY.years_of_history
    )

    df_a_train, df_a_test = train_test_split_by_time(df_a, TRAIN_FRACTION)
    df_b_train, df_b_test = train_test_split_by_time(df_b, TRAIN_FRACTION)
    logger.info(
        "train: %s -> %s (%d velas), test: %s -> %s (%d velas)",
        df_a_train.index[0], df_a_train.index[-1], len(df_a_train),
        df_a_test.index[0], df_a_test.index[-1], len(df_a_test),
    )

    logger.info("Buscando parametros del pairs trade SOLO en TRAIN...")
    best_cfg, train_stats = grid_search_pairs(df_a_train, df_b_train)

    if best_cfg is None:
        report = {
            "status": "no_viable_config",
            "message": "Ninguna combinacion probada da expectancy positiva en train.",
        }
        logger.warning(report["message"])
    else:
        logger.info("Mejor config en TRAIN: %s", asdict(best_cfg))
        logger.info("TRAIN stats: %s", train_stats._asdict())

        _, test_stats = _run(df_a_test, df_b_test, best_cfg)
        holds_up = test_stats.expectancy_pct > 0
        logger.info(
            "TEST stats: %s -> %s", test_stats._asdict(),
            "SOSTIENE fuera de muestra" if holds_up else "NO sostiene fuera de muestra",
        )

        report = {
            "status": "validated" if holds_up else "failed_out_of_sample",
            "config": asdict(best_cfg),
            "train_metrics": train_stats._asdict(),
            "test_metrics": test_stats._asdict(),
            "holds_up_out_of_sample": holds_up,
        }

    out_path = RESULTS_DIR / "pairs_train_test_validation.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte guardado en %s", out_path)


if __name__ == "__main__":
    main()
