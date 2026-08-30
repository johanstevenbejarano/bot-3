"""Walk-forward de breakout/BTC CON filtro de tendencia diaria (EMA50), misma metodología que
`breakout_walk_forward.py` — para comparar directamente si el filtro multi-timeframe mejora el
resultado (-10.58% sin filtro, ver FINDINGS.md) o no.
"""
from __future__ import annotations

import json
import logging

import numpy as np

from src.breakout_validate import grid_search_single_symbol, run_on_raw
from src.config import BACKTEST, RESULTS_DIR
from src.data_fetch import fetch_ohlcv
from src.htf_filter import HTFTrendConfig, add_htf_trend
from src.walk_forward import run_walk_forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SYMBOL = "BTC/USDT"
YEARS_OF_HISTORY = 5
TRAIN_DAYS = 365
TEST_DAYS = 90


def main() -> None:
    raw_1h = fetch_ohlcv(SYMBOL, timeframe=BACKTEST.timeframe, years=YEARS_OF_HISTORY)
    raw = add_htf_trend(raw_1h, SYMBOL, HTFTrendConfig(), years=YEARS_OF_HISTORY)
    logger.info(
        "%s: %d velas de 1h con filtro HTF (%s -> %s), %d velas perdidas por warm-up de la EMA diaria",
        SYMBOL, len(raw), raw.index[0], raw.index[-1], len(raw_1h) - len(raw),
    )

    fold_results = run_walk_forward(
        raw,
        grid_search_fn=lambda train_raw: grid_search_single_symbol(train_raw, min_trades=20),
        run_fn=run_on_raw,
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
    )

    for i, fr in enumerate(fold_results):
        logger.info(
            "Fold %d [train %s->%s | test %s->%s]: train n=%d exp=%.3f%% -> test n=%d exp=%.3f%% ret=%.2f%%",
            i, fr.fold.train_start.date(), fr.fold.train_end.date(),
            fr.fold.test_start.date(), fr.fold.test_end.date(),
            fr.train_num_trades, fr.train_expectancy_pct,
            fr.test_num_trades, fr.test_expectancy_pct, fr.test_return_pct,
        )

    folds_with_trades = [fr for fr in fold_results if fr.test_num_trades > 0]
    total_test_trades = sum(fr.test_num_trades for fr in fold_results)
    positive_folds = sum(1 for fr in folds_with_trades if fr.test_expectancy_pct > 0)

    equity_multiplier = float(np.prod([1 + fr.test_return_pct / 100 for fr in folds_with_trades])) \
        if folds_with_trades else 1.0
    walk_forward_return_pct = (equity_multiplier - 1) * 100

    report = {
        "symbol": SYMBOL,
        "htf_filter": True,
        "train_days": TRAIN_DAYS,
        "test_days": TEST_DAYS,
        "n_folds": len(fold_results),
        "n_folds_with_trades": len(folds_with_trades),
        "n_folds_positive": positive_folds,
        "total_test_trades": total_test_trades,
        "walk_forward_return_pct": walk_forward_return_pct,
        "holds_up": walk_forward_return_pct > 0 and positive_folds > len(folds_with_trades) / 2,
        "folds": [
            {
                "train_start": str(fr.fold.train_start.date()),
                "train_end": str(fr.fold.train_end.date()),
                "test_start": str(fr.fold.test_start.date()),
                "test_end": str(fr.fold.test_end.date()),
                "train_num_trades": fr.train_num_trades,
                "train_expectancy_pct": fr.train_expectancy_pct,
                "test_num_trades": fr.test_num_trades,
                "test_expectancy_pct": fr.test_expectancy_pct,
                "test_return_pct": fr.test_return_pct,
            }
            for fr in fold_results
        ],
    }

    out_path = RESULTS_DIR / "breakout_htf_walk_forward_BTC.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info(
        "Walk-forward CON filtro HTF: %.2f%% acumulado sobre %d folds (%d con trades, %d positivos, %d trades totales)",
        walk_forward_return_pct, len(fold_results), len(folds_with_trades), positive_folds, total_test_trades,
    )
    logger.info("Reporte guardado en %s", out_path)


if __name__ == "__main__":
    main()
