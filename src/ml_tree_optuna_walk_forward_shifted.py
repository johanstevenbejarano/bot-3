"""Prueba de robustez del resultado de `ml_tree_optuna_walk_forward.py` (+0.44% acumulado, 7/14
folds positivos): misma metodología exacta (Optuna + CV purgada + train de 2 años sobre 9 años de
historia), pero con el origen de los folds desplazado ~75 días — sin tocar ningún hiperparámetro
de búsqueda. Si el resultado sigue rondando cero (positivo o negativo, no importa el signo
exacto), confirma que es ruido alrededor del punto de equilibrio. Si se derrumba a algo
claramente negativo, confirma que el resultado original era sensible a una elección arbitraria
(dónde arranca el primer fold), no una señal real.

Logea el progreso apenas termina cada fold (a diferencia del script original, que solo
imprimía todo al final) para poder seguir la ejecución en vivo.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.config import BACKTEST, ML_STRATEGY, RESULTS_DIR
from src.data_fetch import fetch_ohlcv
from src.ml_features import compute_features
from src.ml_strategy import ModelBundle, RunResult, prepare_labeled, simulate
from src.ml_tree_strategy import train_tree_classifiers_optuna
from src.walk_forward import FoldResult, run_walk_forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SYMBOL = "BTC/USDT"
YEARS_OF_HISTORY = 9
TRAIN_DAYS = 730
TEST_DAYS = 90
STEP_DAYS = 180
N_TRIALS = 25
N_CV_SPLITS = 3
SHIFT_DAYS = 75  # desplaza el origen de los folds -- la única diferencia con la corrida original


def _grid_search_fn(train_slice: pd.DataFrame) -> ModelBundle | None:
    labeled = prepare_labeled(train_slice, ML_STRATEGY)
    if labeled.empty:
        return None
    return train_tree_classifiers_optuna(labeled, ML_STRATEGY, n_trials=N_TRIALS, n_cv_splits=N_CV_SPLITS)


def _run_fn(df_slice: pd.DataFrame, bundle: ModelBundle | None) -> RunResult:
    if bundle is None:
        return RunResult(0, 0.0, 0.0, 0.0, 0.0, float("nan"))
    labeled = prepare_labeled(df_slice, ML_STRATEGY)
    if labeled.empty:
        return RunResult(0, 0.0, 0.0, 0.0, 0.0, float("nan"))
    return simulate(labeled, bundle, ML_STRATEGY)


def _log_fold_progress(idx: int, total: int, fr: FoldResult) -> None:
    logger.info(
        "[%d/%d] fold [train %s->%s | test %s->%s]: train n=%d exp=%.3f%% -> test n=%d exp=%.3f%% ret=%.2f%%",
        idx + 1, total,
        fr.fold.train_start.date(), fr.fold.train_end.date(),
        fr.fold.test_start.date(), fr.fold.test_end.date(),
        fr.train_num_trades, fr.train_expectancy_pct,
        fr.test_num_trades, fr.test_expectancy_pct, fr.test_return_pct,
    )


def main() -> None:
    raw = fetch_ohlcv(SYMBOL, timeframe=BACKTEST.timeframe, years=YEARS_OF_HISTORY)
    features = compute_features(raw)

    shifted_start = features.index[0] + pd.Timedelta(days=SHIFT_DAYS)
    features = features.loc[shifted_start:]

    logger.info(
        "%s: %d velas con features, origen desplazado %d dias (%s -> %s)",
        SYMBOL, len(features), SHIFT_DAYS, features.index[0], features.index[-1],
    )

    fold_results = run_walk_forward(
        features,
        grid_search_fn=_grid_search_fn,
        run_fn=_run_fn,
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        step_days=STEP_DAYS,
        on_fold_complete=_log_fold_progress,
    )

    folds_with_trades = [fr for fr in fold_results if fr.test_num_trades > 0]
    total_test_trades = sum(fr.test_num_trades for fr in fold_results)
    positive_folds = sum(1 for fr in folds_with_trades if fr.test_expectancy_pct > 0)

    equity_multiplier = float(np.prod([1 + fr.test_return_pct / 100 for fr in folds_with_trades])) \
        if folds_with_trades else 1.0
    walk_forward_return_pct = (equity_multiplier - 1) * 100

    report = {
        "symbol": SYMBOL,
        "strategy": "ml_tree_optuna SHIFTED +75 dias (prueba de robustez)",
        "shift_days": SHIFT_DAYS,
        "years_of_history": YEARS_OF_HISTORY,
        "train_days": TRAIN_DAYS,
        "test_days": TEST_DAYS,
        "step_days": STEP_DAYS,
        "n_trials": N_TRIALS,
        "n_cv_splits": N_CV_SPLITS,
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

    out_path = RESULTS_DIR / "ml_tree_optuna_walk_forward_BTC_shifted.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info(
        "RESULTADO FINAL (shifted +%dd): %.2f%% acumulado sobre %d folds (%d con trades, %d positivos, %d trades totales)",
        SHIFT_DAYS, walk_forward_return_pct, len(fold_results), len(folds_with_trades), positive_folds, total_test_trades,
    )
    logger.info("Reporte guardado en %s", out_path)


if __name__ == "__main__":
    main()
