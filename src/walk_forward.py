"""Motor genérico de validación walk-forward: recalibra periódicamente en vez de un solo split
train/test.

Un split 70/30 fijo asume que los parámetros encontrados en el pasado siguen sirviendo para
siempre. Walk-forward simula algo más realista: en cada "fold" se buscan parámetros usando solo
una ventana móvil de datos pasados, se aplican en el tramo siguiente nunca visto, y se avanza. Al
concatenar todos los tramos de test se obtiene una curva "fuera de muestra" continua, mucho más
exigente que un solo split (y mucho más cara de correr, porque repite la búsqueda de parámetros
una vez por fold).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Protocol

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(
    index: pd.DatetimeIndex, train_days: int, test_days: int, step_days: int | None = None
) -> list[WalkForwardFold]:
    """Genera folds [train_start, train_end) -> [test_start, test_end) deslizando por step_days."""
    step_days = step_days or test_days
    start, end = index[0], index[-1]

    folds = []
    train_start = start
    while True:
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_days)
        if test_end > end:
            break
        folds.append(WalkForwardFold(train_start, train_end, test_start, test_end))
        train_start = train_start + pd.Timedelta(days=step_days)

    return folds


class RunResultLike(Protocol):
    num_trades: int
    expectancy_pct: float
    return_pct: float


class FoldResult(NamedTuple):
    fold: WalkForwardFold
    train_num_trades: int
    train_expectancy_pct: float
    test_num_trades: int
    test_expectancy_pct: float
    test_return_pct: float


def run_walk_forward(
    raw: pd.DataFrame,
    grid_search_fn: Callable[[pd.DataFrame], object | None],
    run_fn: Callable[[pd.DataFrame, object], RunResultLike],
    train_days: int = 365,
    test_days: int = 90,
    step_days: int | None = None,
    on_fold_complete: Callable[[int, int, FoldResult], None] | None = None,
) -> list[FoldResult]:
    """Para cada fold: busca config SOLO con `train_days` de historia pasada (grid_search_fn),
    la aplica sin retocar sobre los `test_days` siguientes nunca vistos (run_fn), y avanza.

    `on_fold_complete(idx, total, fold_result)`, si se pasa, se llama apenas termina cada fold —
    para poder loguear/imprimir progreso incremental en corridas largas en vez de esperar a que
    terminen todos los folds para ver cualquier resultado.
    """
    folds = make_folds(raw.index, train_days, test_days, step_days)
    results = []

    for idx, f in enumerate(folds):
        train_raw = raw.loc[f.train_start:f.train_end]
        test_raw = raw.loc[f.test_start:f.test_end]

        cfg = grid_search_fn(train_raw)
        if cfg is None:
            fold_result = FoldResult(f, 0, 0.0, 0, 0.0, 0.0)
        else:
            train_result = run_fn(train_raw, cfg)
            test_result = run_fn(test_raw, cfg)
            fold_result = FoldResult(
                fold=f,
                train_num_trades=train_result.num_trades,
                train_expectancy_pct=train_result.expectancy_pct,
                test_num_trades=test_result.num_trades,
                test_expectancy_pct=test_result.expectancy_pct,
                test_return_pct=test_result.return_pct,
            )

        results.append(fold_result)
        if on_fold_complete:
            on_fold_complete(idx, len(folds), fold_result)

    return results
