"""Walk-forward de la estrategia de breakout sobre BTC/USDT: recalibra cada `test_days` usando
los `train_days` previos, evalúa en el tramo siguiente nunca visto, y concatena todos los tramos
de test en una curva de equity "fuera de muestra" continua.

Se eligió breakout/BTC porque fue el mejor resultado de TRAIN de toda la sesión en un solo split
(Sharpe 0.855, ver FINDINGS.md "Chequeo BTC-solo") — el candidato más prometedor para ver si la
recalibración periódica sostiene la ventaja a través de distintos regímenes de mercado.
"""
from __future__ import annotations

import json
import logging

import numpy as np

from src.breakout_validate import grid_search_single_symbol, run_on_raw
from src.config import BACKTEST, RESULTS_DIR
from src.data_fetch import fetch_ohlcv
from src.walk_forward import run_walk_forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SYMBOL = "BTC/USDT"
YEARS_OF_HISTORY = 5
TRAIN_DAYS = 365
TEST_DAYS = 90


def main() -> None:
    raw = fetch_ohlcv(SYMBOL, timeframe=BACKTEST.timeframe, years=YEARS_OF_HISTORY)
    logger.info("%s: %d velas (%s -> %s)", SYMBOL, len(raw), raw.index[0], raw.index[-1])

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
    folds_with_cfg = [fr for fr in fold_results if fr.train_num_trades > 0 or fr.test_num_trades > 0]
    total_test_trades = sum(fr.test_num_trades for fr in fold_results)
    positive_folds = sum(1 for fr in folds_with_trades if fr.test_expectancy_pct > 0)

    # curva de equity walk-forward: multiplica el efecto de cada tramo de test en secuencia
    equity_multiplier = float(np.prod([1 + fr.test_return_pct / 100 for fr in folds_with_trades])) \
        if folds_with_trades else 1.0
    walk_forward_return_pct = (equity_multiplier - 1) * 100

    report = {
        "symbol": SYMBOL,
        "train_days": TRAIN_DAYS,
        "test_days": TEST_DAYS,
        "n_folds": len(fold_results),
        "n_folds_with_config": len(folds_with_cfg),
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

    out_path = RESULTS_DIR / "breakout_walk_forward_BTC.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info(
        "Walk-forward: %.2f%% acumulado sobre %d folds (%d con trades, %d positivos, %d trades totales)",
        walk_forward_return_pct, len(fold_results), len(folds_with_trades), positive_folds, total_test_trades,
    )
    logger.info("Reporte guardado en %s", out_path)


if __name__ == "__main__":
    main()
