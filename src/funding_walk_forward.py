"""Walk-forward de la estrategia contraria de funding sobre BTC/USDT: misma metodología que
`breakout_walk_forward.py` (16 folds, recalibra cada 90 días con los 365 previos)."""
from __future__ import annotations

import json
import logging

import numpy as np

from src.config import BACKTEST, FUNDING_STRATEGY, RESULTS_DIR
from src.data_fetch import fetch_ohlcv
from src.funding_indicators import add_funding_indicators
from src.funding_validate import grid_search_single_symbol, run_config
from src.walk_forward import run_walk_forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SYMBOL = "BTC/USDT"
YEARS_OF_HISTORY = 5
TRAIN_DAYS = 365
TEST_DAYS = 90


def main() -> None:
    raw_1h = fetch_ohlcv(SYMBOL, timeframe=BACKTEST.timeframe, years=YEARS_OF_HISTORY)
    raw = add_funding_indicators(raw_1h, SYMBOL, FUNDING_STRATEGY, years=YEARS_OF_HISTORY)
    logger.info("%s: %d velas con funding mergeado (%s -> %s)", SYMBOL, len(raw), raw.index[0], raw.index[-1])

    fold_results = run_walk_forward(
        raw,
        grid_search_fn=lambda train_raw: grid_search_single_symbol(train_raw, min_trades=15),
        run_fn=run_config,
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
        "strategy": "funding_contrarian",
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

    out_path = RESULTS_DIR / "funding_walk_forward_BTC.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info(
        "Walk-forward funding: %.2f%% acumulado sobre %d folds (%d con trades, %d positivos, %d trades totales)",
        walk_forward_return_pct, len(fold_results), len(folds_with_trades), positive_folds, total_test_trades,
    )
    logger.info("Reporte guardado en %s", out_path)


if __name__ == "__main__":
    main()
