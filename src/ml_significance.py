"""Intervalo de confianza bootstrap sobre retornos trade-por-trade.

Que el agregado de un walk-forward dé positivo no dice si esa cifra es distinguible de ruido. El
bootstrap remuestrea (con reemplazo) los PnL% de trades individuales muchas veces y mide qué tan
estable es la expectancy resultante — si el intervalo de confianza no incluye cero, hay evidencia
estadística de que el promedio no es casualidad; si lo incluye, no se puede descartar el azar.
"""
from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    values: list[float] | np.ndarray, n_boot: int = 5000, ci: float = 0.95, seed: int = 42
) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "excludes_zero": False}

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = (1 - ci) / 2
    ci_low = float(np.percentile(boot_means, alpha * 100))
    ci_high = float(np.percentile(boot_means, (1 - alpha) * 100))

    return {
        "n": n,
        "mean": float(values.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "excludes_zero": ci_low > 0 or ci_high < 0,
        "n_boot": n_boot,
        "confidence": ci,
    }
