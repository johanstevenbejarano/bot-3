"""Tercera verificación, extensa y estadística: en vez de otro desplazamiento de folds sobre
BTC (que ya empezaba a parecerse a "buscar hasta que confirme"), se prueban dos cosas genuinamente
distintas de lo ya hecho:

1. La MISMA metodología exacta (Optuna + CV purgada + train de 2 años sobre 9 años de historia,
   sin origen desplazado) aplicada a ETH/USDT — un activo independiente, no otro recorte de los
   mismos datos de BTC.
2. Un intervalo de confianza bootstrap sobre los retornos trade-por-trade de cada símbolo: si el
   intervalo no incluye cero, hay evidencia estadística de que la expectancy no es casualidad: si
   lo incluye, no se puede descartar el azar (independientemente de que el agregado dé positivo).
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.config import BACKTEST, ML_STRATEGY, RESULTS_DIR
from src.data_fetch import fetch_ohlcv
from src.ml_features import compute_features
from src.ml_significance import bootstrap_mean_ci
from src.ml_strategy import ModelBundle, RunResult, prepare_labeled, simulate_with_trades
from src.ml_tree_strategy import train_tree_classifiers_optuna
from src.walk_forward import FoldResult, run_walk_forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SYMBOLS = ("BTC/USDT", "ETH/USDT")
YEARS_OF_HISTORY = 9
TRAIN_DAYS = 730
TEST_DAYS = 90
STEP_DAYS = 180
N_TRIALS = 25
N_CV_SPLITS = 3


def _grid_search_fn(train_slice: pd.DataFrame) -> ModelBundle | None:
    labeled = prepare_labeled(train_slice, ML_STRATEGY)
    if labeled.empty:
        return None
    return train_tree_classifiers_optuna(labeled, ML_STRATEGY, n_trials=N_TRIALS, n_cv_splits=N_CV_SPLITS)


def _log_fold_progress(idx: int, total: int, fr: FoldResult) -> None:
    logger.info(
        "[%d/%d] fold [train %s->%s | test %s->%s]: train n=%d exp=%.3f%% -> test n=%d exp=%.3f%% ret=%.2f%%",
        idx + 1, total,
        fr.fold.train_start.date(), fr.fold.train_end.date(),
        fr.fold.test_start.date(), fr.fold.test_end.date(),
        fr.train_num_trades, fr.train_expectancy_pct,
        fr.test_num_trades, fr.test_expectancy_pct, fr.test_return_pct,
    )


def run_for_symbol(symbol: str) -> dict:
    raw = fetch_ohlcv(symbol, timeframe=BACKTEST.timeframe, years=YEARS_OF_HISTORY)
    features = compute_features(raw)
    logger.info(
        "%s: %d velas con features (%s -> %s)", symbol, len(features), features.index[0], features.index[-1]
    )

    all_test_trades: list[float] = []

    # run_walk_forward llama run_fn(train_raw, cfg) y LUEGO run_fn(test_raw, cfg) en cada fold
    # (ver src/walk_forward.py) — con ese orden fijo, un contador par/impar basta para saber cuál
    # de las dos llamadas es la de test, sin tener que comparar los DataFrames entre sí.
    call_counter = {"n": 0}

    def run_fn(df_slice: pd.DataFrame, bundle: ModelBundle | None) -> RunResult:
        call_counter["n"] += 1
        is_test_call = call_counter["n"] % 2 == 0  # 1ra llamada del fold = train, 2da = test

        if bundle is None:
            return RunResult(0, 0.0, 0.0, 0.0, 0.0, float("nan"))
        labeled = prepare_labeled(df_slice, ML_STRATEGY)
        if labeled.empty:
            return RunResult(0, 0.0, 0.0, 0.0, 0.0, float("nan"))

        result, trade_pnls = simulate_with_trades(labeled, bundle, ML_STRATEGY)
        if is_test_call:
            all_test_trades.extend(trade_pnls)
        return result

    fold_results = run_walk_forward(
        features,
        grid_search_fn=_grid_search_fn,
        run_fn=run_fn,
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        step_days=STEP_DAYS,
        on_fold_complete=_log_fold_progress,
    )

    folds_with_trades = [fr for fr in fold_results if fr.test_num_trades > 0]
    positive_folds = sum(1 for fr in folds_with_trades if fr.test_expectancy_pct > 0)
    equity_multiplier = float(np.prod([1 + fr.test_return_pct / 100 for fr in folds_with_trades])) \
        if folds_with_trades else 1.0
    walk_forward_return_pct = (equity_multiplier - 1) * 100

    sig = bootstrap_mean_ci(all_test_trades, n_boot=5000)

    logger.info(
        "%s: walk-forward %.2f%% acumulado (%d/%d folds positivos, %d trades) | bootstrap media=%.4f%% "
        "IC95%%=[%.4f%%, %.4f%%] excluye_cero=%s",
        symbol, walk_forward_return_pct, positive_folds, len(folds_with_trades), len(all_test_trades),
        sig["mean"], sig["ci_low"], sig["ci_high"], sig["excludes_zero"],
    )

    return {
        "symbol": symbol,
        "n_folds": len(fold_results),
        "n_folds_with_trades": len(folds_with_trades),
        "n_folds_positive": positive_folds,
        "total_test_trades": len(all_test_trades),
        "walk_forward_return_pct": walk_forward_return_pct,
        "bootstrap": sig,
    }


def main() -> None:
    report = {symbol: run_for_symbol(symbol) for symbol in SYMBOLS}

    out_path = RESULTS_DIR / "ml_tree_optuna_significance_check.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte guardado en %s", out_path)


if __name__ == "__main__":
    main()
