"""Simulación de Monte Carlo: reordena los trades del backtest para estimar el rango de drawdown."""
from __future__ import annotations

import numpy as np


def _max_drawdown_from_returns(returns_frac: np.ndarray) -> float:
    equity = np.cumprod(1 + returns_frac)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min() * 100)  # % (valor negativo)


def monte_carlo_drawdown(
    trade_returns_pct: np.ndarray, n_iterations: int = 2000, seed: int | None = None
) -> dict:
    """Reordena aleatoriamente la secuencia real de trades N veces y reporta percentiles de drawdown."""
    rng = np.random.default_rng(seed)
    returns_frac = np.asarray(trade_returns_pct, dtype=float) / 100.0

    drawdowns = np.empty(n_iterations)
    for i in range(n_iterations):
        shuffled = rng.permutation(returns_frac)
        drawdowns[i] = _max_drawdown_from_returns(shuffled)

    return {
        "p5": float(np.percentile(drawdowns, 5)),
        "p50": float(np.percentile(drawdowns, 50)),
        "p95": float(np.percentile(drawdowns, 95)),
        "worst": float(drawdowns.min()),
        "n_iterations": n_iterations,
    }
