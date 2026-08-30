"""Entrena los clasificadores (long/short) sobre el set de features + labels triple-barrera, y
simula la estrategia resultante con un loop event-driven (sin posiciones superpuestas, igual que
todas las estrategias de backtesting.py de la sesión).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import BACKTEST, MLConfig
from src.ml_features import FEATURE_COLUMNS
from src.ml_labels import triple_barrier_labels


@dataclass
class ModelBundle:
    scaler: StandardScaler
    model_long: LogisticRegression
    model_short: LogisticRegression
    threshold_long: float
    threshold_short: float


class RunResult(NamedTuple):
    num_trades: int
    win_rate_pct: float
    expectancy_pct: float
    return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float


def prepare_labeled(df_with_features: pd.DataFrame, cfg: MLConfig) -> pd.DataFrame:
    """Aplica el labeling triple-barrera sobre un DataFrame que ya tiene las features calculadas
    (y la columna `atr`, que ml_features.compute_features también deja disponible)."""
    return triple_barrier_labels(
        df_with_features, cfg.sl_atr_mult, cfg.tp_atr_mult, cfg.max_holding_bars,
        commission=BACKTEST.commission,
    )


def train_classifiers(labeled_df: pd.DataFrame, cfg: MLConfig) -> ModelBundle | None:
    if len(labeled_df) < cfg.min_train_samples:
        return None

    X = labeled_df[FEATURE_COLUMNS].to_numpy()
    y_long = (labeled_df["long_pnl_pct"] > 0).to_numpy()
    y_short = (labeled_df["short_pnl_pct"] > 0).to_numpy()

    # si una clase no tiene ambos valores (todo gana o todo pierde), no se puede entrenar
    if y_long.sum() < 5 or (~y_long).sum() < 5 or y_short.sum() < 5 or (~y_short).sum() < 5:
        return None

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    model_long = LogisticRegression(C=0.5, max_iter=1000).fit(X_scaled, y_long)
    model_short = LogisticRegression(C=0.5, max_iter=1000).fit(X_scaled, y_short)

    # Umbral relativo: percentil de las probabilidades predichas EN EL PROPIO TRAIN (nunca en
    # test) — así el modelo se evalúa contra su propio extremo de mayor confianza en vez de un
    # número fijo que podría ser inalcanzable dada la tasa base de acierto.
    train_prob_long = model_long.predict_proba(X_scaled)[:, 1]
    train_prob_short = model_short.predict_proba(X_scaled)[:, 1]
    threshold_long = float(np.quantile(train_prob_long, 1 - cfg.top_fraction))
    threshold_short = float(np.quantile(train_prob_short, 1 - cfg.top_fraction))

    return ModelBundle(
        scaler=scaler, model_long=model_long, model_short=model_short,
        threshold_long=threshold_long, threshold_short=threshold_short,
    )


def _collect_trade_pnls(labeled_df: pd.DataFrame, bundle: ModelBundle) -> list[float]:
    X = labeled_df[FEATURE_COLUMNS].to_numpy()
    X_scaled = bundle.scaler.transform(X)
    prob_long = bundle.model_long.predict_proba(X_scaled)[:, 1]
    prob_short = bundle.model_short.predict_proba(X_scaled)[:, 1]

    long_pnl = labeled_df["long_pnl_pct"].to_numpy()
    long_offset = labeled_df["long_exit_offset"].to_numpy()
    short_pnl = labeled_df["short_pnl_pct"].to_numpy()
    short_offset = labeled_df["short_exit_offset"].to_numpy()

    n = len(labeled_df)
    trade_pnls: list[float] = []

    i = 0
    while i < n:
        take_long = prob_long[i] >= bundle.threshold_long and prob_long[i] > prob_short[i]
        take_short = prob_short[i] >= bundle.threshold_short and prob_short[i] > prob_long[i]

        if take_long:
            trade_pnls.append(long_pnl[i])
            i += max(int(long_offset[i]), 1)
        elif take_short:
            trade_pnls.append(short_pnl[i])
            i += max(int(short_offset[i]), 1)
        else:
            i += 1

    return trade_pnls


def simulate(labeled_df: pd.DataFrame, bundle: ModelBundle, cfg: MLConfig) -> RunResult:
    """Recorre las barras en orden; cuando el modelo predice long o short con probabilidad por
    encima del umbral, 'toma' el trade cuyo resultado ya está precalculado por el labeling
    triple-barrera, y salta hasta que ese trade se resuelve (sin posiciones superpuestas)."""
    return simulate_with_trades(labeled_df, bundle, cfg)[0]


def simulate_with_trades(
    labeled_df: pd.DataFrame, bundle: ModelBundle, cfg: MLConfig
) -> tuple[RunResult, list[float]]:
    """Igual que `simulate`, pero además devuelve la lista de PnL% de cada trade individual —
    para análisis estadístico (ej. bootstrap) que necesita los datos crudos, no solo el resumen.
    """
    trade_pnls = _collect_trade_pnls(labeled_df, bundle)

    n_trades = len(trade_pnls)
    if n_trades == 0:
        return RunResult(0, 0.0, 0.0, 0.0, 0.0, float("nan")), []

    returns_frac = np.array(trade_pnls) / 100
    equity_curve = np.cumprod(1 + cfg.risk_per_trade * returns_frac)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max

    result = RunResult(
        num_trades=n_trades,
        win_rate_pct=float((returns_frac > 0).mean() * 100),
        expectancy_pct=float(returns_frac.mean() * 100),
        return_pct=float((equity_curve[-1] - 1) * 100),
        max_drawdown_pct=float(drawdown.min() * 100),
        sharpe_ratio=float(returns_frac.mean() / returns_frac.std()) if returns_frac.std() > 0 else float("nan"),
    )
    return result, trade_pnls
