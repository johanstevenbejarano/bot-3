"""Pairs trading BTC/ETH: spread, z-score, y motor de backtest a medida.

`backtesting.py` está pensado para un solo activo por Backtest; una operación de pairs trading
mueve dos posiciones simultáneas (larga en una moneda, corta en la otra) sobre el mismo evento,
así que el motor de backtest aquí es un loop simple y explícito en vez de reutilizar esa librería.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from src.config import BACKTEST, PairsConfig


def compute_spread_zscore(df_a: pd.DataFrame, df_b: pd.DataFrame, cfg: PairsConfig) -> pd.DataFrame:
    """Alinea ambas series por timestamp y calcula el spread (log_a - beta*log_b) y su z-score.

    `beta` se estima con una regresión rodante (cov/var, solo con datos pasados — sin lookahead)
    para no asumir una relación 1:1 fija entre BTC y ETH.
    """
    aligned = pd.DataFrame({"close_a": df_a["close"], "close_b": df_b["close"]}).dropna()

    log_a = np.log(aligned["close_a"])
    log_b = np.log(aligned["close_b"])

    cov = log_a.rolling(cfg.beta_window).cov(log_b)
    var_b = log_b.rolling(cfg.beta_window).var()
    beta = cov / var_b

    spread = log_a - beta * log_b
    mean_spread = spread.rolling(cfg.z_window).mean()
    std_spread = spread.rolling(cfg.z_window).std()
    zscore = (spread - mean_spread) / std_spread

    out = aligned.copy()
    out["beta"] = beta
    out["spread"] = spread
    out["zscore"] = zscore
    return out.dropna()


class PairsRunResult(NamedTuple):
    num_trades: int
    win_rate_pct: float
    expectancy_pct: float
    return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float  # por trade, NO anualizado (no comparable directo con backtesting.py)


def backtest_pairs(
    df: pd.DataFrame, cfg: PairsConfig, commission: float = BACKTEST.commission
) -> tuple[pd.DataFrame, PairsRunResult]:
    """Simula el pairs trade: abre cuando |z|>=entry_z, cierra en reversión (|z|<=exit_z) o en
    stop (|z|>=stop_z, la divergencia siguió creciendo en vez de revertir).

    Ambas patas se dimensionan a notional igual (dollar-neutral). El PnL de cada trade es el
    spread de retornos entre las dos patas menos 4 comisiones (entrada+salida en cada pata) —
    una aproximación simple, documentada, no un modelo de slippage/margen real.
    """
    close_a = df["close_a"].to_numpy()
    close_b = df["close_b"].to_numpy()
    z = df["zscore"].to_numpy()
    index = df.index

    state = 0  # 0 = flat, 1 = long A / short B, -1 = short A / long B
    entry_i = 0
    trades = []

    for i in range(len(df)):
        if state == 0:
            if z[i] <= -cfg.entry_z:
                state, entry_i = 1, i
            elif z[i] >= cfg.entry_z:
                state, entry_i = -1, i
            continue

        if state == 1:
            exit_reason = "reverted" if z[i] >= -cfg.exit_z else ("stopped" if z[i] <= -cfg.stop_z else None)
        else:
            exit_reason = "reverted" if z[i] <= cfg.exit_z else ("stopped" if z[i] >= cfg.stop_z else None)

        if exit_reason is None:
            continue

        ret_a = close_a[i] / close_a[entry_i] - 1
        ret_b = close_b[i] / close_b[entry_i] - 1
        gross_pnl_pct = (ret_a - ret_b) if state == 1 else (ret_b - ret_a)
        net_pnl_pct = gross_pnl_pct - 4 * commission

        trades.append({
            "entry_time": index[entry_i],
            "exit_time": index[i],
            "direction": "long_a_short_b" if state == 1 else "short_a_long_b",
            "entry_z": z[entry_i],
            "exit_z": z[i],
            "gross_pnl_pct": gross_pnl_pct * 100,
            "net_pnl_pct": net_pnl_pct * 100,
            "exit_reason": exit_reason,
        })
        state = 0

    trades_df = pd.DataFrame(trades)
    n = len(trades_df)

    if n == 0:
        return trades_df, PairsRunResult(0, 0.0, 0.0, 0.0, 0.0, float("nan"))

    returns_frac = trades_df["net_pnl_pct"].to_numpy() / 100
    equity_curve = np.cumprod(1 + cfg.risk_per_trade * returns_frac)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max

    stats = PairsRunResult(
        num_trades=n,
        win_rate_pct=float((trades_df["net_pnl_pct"] > 0).mean() * 100),
        expectancy_pct=float(trades_df["net_pnl_pct"].mean()),
        return_pct=float((equity_curve[-1] - 1) * 100),
        max_drawdown_pct=float(drawdown.min() * 100),
        sharpe_ratio=float(returns_frac.mean() / returns_frac.std()) if returns_frac.std() > 0 else float("nan"),
    )
    return trades_df, stats
