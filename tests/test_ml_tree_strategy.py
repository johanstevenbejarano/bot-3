import numpy as np
import pandas as pd

from src.config import MLConfig
from src.ml_features import FEATURE_COLUMNS
from src.ml_tree_strategy import train_tree_classifiers, train_tree_classifiers_optuna


def _synthetic_labeled(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    # una señal real y débil en una sola feature, para que haya algo que el modelo pueda aprender
    signal = data["rsi"]
    data["long_pnl_pct"] = signal * 0.5 + rng.normal(scale=2.0, size=n)
    data["short_pnl_pct"] = -signal * 0.5 + rng.normal(scale=2.0, size=n)
    data["long_exit_offset"] = np.ones(n, dtype=int)
    data["short_exit_offset"] = np.ones(n, dtype=int)
    return pd.DataFrame(data, index=index)


def test_train_tree_classifiers_returns_none_with_too_few_samples():
    df = _synthetic_labeled(n=50)
    result = train_tree_classifiers(df, MLConfig(min_train_samples=200))
    assert result is None


def test_train_tree_classifiers_succeeds_with_enough_samples():
    df = _synthetic_labeled(n=1500)
    result = train_tree_classifiers(df, MLConfig(min_train_samples=200), min_val_trades=5)

    assert result is not None
    assert 0.0 <= result.threshold_long <= 1.0
    assert 0.0 <= result.threshold_short <= 1.0


def test_final_bundle_is_refit_on_full_train_not_just_the_fit_slice():
    # el modelo final debe haber visto TODAS las filas de train_labeled (fit + validación
    # interna) al ajustarse -- lo verificamos indirectamente: el scaler final debe reflejar la
    # media de todo el set, no solo la del primer 80%.
    df = _synthetic_labeled(n=1500)
    result = train_tree_classifiers(df, MLConfig(min_train_samples=200), min_val_trades=5)
    assert result is not None

    full_mean = df[FEATURE_COLUMNS].to_numpy().mean(axis=0)
    fit_only_mean = df[FEATURE_COLUMNS].to_numpy()[: int(len(df) * 0.8)].mean(axis=0)

    assert np.allclose(result.scaler.mean_, full_mean)
    assert not np.allclose(result.scaler.mean_, fit_only_mean)


def test_optuna_search_returns_none_with_too_few_samples_for_cv():
    df = _synthetic_labeled(n=50)
    result = train_tree_classifiers_optuna(df, MLConfig(min_train_samples=200), n_trials=5, n_cv_splits=3)
    assert result is None


def test_optuna_search_succeeds_with_enough_samples():
    df = _synthetic_labeled(n=2000, seed=1)
    result = train_tree_classifiers_optuna(
        df, MLConfig(min_train_samples=200, max_holding_bars=5), n_trials=5, n_cv_splits=3, min_val_trades=3,
    )
    assert result is not None
    assert 0.0 <= result.threshold_long <= 1.0
