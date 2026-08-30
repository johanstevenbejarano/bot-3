"""Validación cruzada purgada (con embargo) para series de tiempo.

Un k-fold normal en datos financieros con labeling triple-barrera puede filtrar información: si
una fila de train está justo antes del corte de validación, su etiqueta mira hasta
`max_holding_bars` adelante — pudiendo "ver" datos que ya son parte de la validación. El embargo
elimina ese margen entre el final del train y el inicio de la validación en cada pliegue.
"""
from __future__ import annotations

import numpy as np


def purged_time_series_splits(
    n: int, n_splits: int, embargo: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Genera `n_splits` pliegues secuenciales tipo walk-forward: cada uno entrena con todo lo
    anterior al pliegue de validación, dejando un hueco de `embargo` barras antes de validar.
    """
    val_size = n // (n_splits + 1)
    splits = []

    for k in range(1, n_splits + 1):
        train_end = val_size * k
        val_start = train_end + embargo
        val_end = min(val_start + val_size, n)
        if train_end <= 0 or val_start >= val_end:
            continue
        splits.append((np.arange(0, train_end), np.arange(val_start, val_end)))

    return splits
