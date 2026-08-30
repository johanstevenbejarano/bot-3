import numpy as np
import pandas as pd

from src.config import MLConfig
from src.ml_features import FEATURE_COLUMNS
from src.ml_strategy import ModelBundle, simulate, simulate_with_trades, train_classifiers


class _FakeScaler:
    def transform(self, X):
        return X


class _FakeModel:
    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self.probs, self.probs])


def _labeled_df(n: int, long_offset_first: int) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    data = {col: [0.0] * n for col in FEATURE_COLUMNS}
    data["long_pnl_pct"] = [5.0] + [0.0] * (n - 1)
    data["long_exit_offset"] = [long_offset_first] + [1] * (n - 1)
    data["short_pnl_pct"] = [0.0] * n
    data["short_exit_offset"] = [1] * n
    return pd.DataFrame(data, index=index)


def test_simulate_takes_one_trade_and_skips_bars_until_it_resolves():
    df = _labeled_df(n=5, long_offset_first=3)
    bundle = ModelBundle(
        scaler=_FakeScaler(),
        model_long=_FakeModel([0.9, 0.9, 0.1, 0.1, 0.1]),
        model_short=_FakeModel([0.1, 0.1, 0.1, 0.1, 0.1]),
        threshold_long=0.55, threshold_short=0.55,
    )
    cfg = MLConfig(risk_per_trade=0.02)

    result = simulate(df, bundle, cfg)

    assert result.num_trades == 1  # solo el trade en la barra 0, no vuelve a entrar en la barra 1
    assert result.expectancy_pct == 5.0


def test_simulate_with_trades_returns_same_result_plus_raw_pnls():
    df = _labeled_df(n=5, long_offset_first=3)
    bundle = ModelBundle(
        scaler=_FakeScaler(),
        model_long=_FakeModel([0.9, 0.9, 0.1, 0.1, 0.1]),
        model_short=_FakeModel([0.1, 0.1, 0.1, 0.1, 0.1]),
        threshold_long=0.55, threshold_short=0.55,
    )
    cfg = MLConfig(risk_per_trade=0.02)

    result, trade_pnls = simulate_with_trades(df, bundle, cfg)
    direct_result = simulate(df, bundle, cfg)

    assert result.num_trades == direct_result.num_trades
    assert result.expectancy_pct == direct_result.expectancy_pct
    assert result.return_pct == direct_result.return_pct
    assert trade_pnls == [5.0]


def test_simulate_no_trade_when_neither_side_clears_threshold():
    df = _labeled_df(n=3, long_offset_first=1)
    bundle = ModelBundle(
        scaler=_FakeScaler(),
        model_long=_FakeModel([0.5, 0.5, 0.5]),
        model_short=_FakeModel([0.5, 0.5, 0.5]),
        threshold_long=0.55, threshold_short=0.55,
    )
    cfg = MLConfig()

    result = simulate(df, bundle, cfg)

    assert result.num_trades == 0


def test_simulate_no_trade_on_exact_tie_between_long_and_short():
    df = _labeled_df(n=3, long_offset_first=1)
    bundle = ModelBundle(
        scaler=_FakeScaler(),
        model_long=_FakeModel([0.9, 0.9, 0.9]),
        model_short=_FakeModel([0.9, 0.9, 0.9]),  # empate exacto -> ninguno "gana"
        threshold_long=0.55, threshold_short=0.55,
    )
    cfg = MLConfig()

    result = simulate(df, bundle, cfg)

    assert result.num_trades == 0


def test_train_classifiers_returns_none_with_too_few_samples():
    index = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    data = {col: np.random.default_rng(0).normal(size=10) for col in FEATURE_COLUMNS}
    data["long_pnl_pct"] = [1.0, -1.0] * 5
    data["short_pnl_pct"] = [-1.0, 1.0] * 5
    df = pd.DataFrame(data, index=index)

    result = train_classifiers(df, MLConfig(min_train_samples=200))

    assert result is None


def test_train_classifiers_succeeds_with_enough_varied_samples():
    rng = np.random.default_rng(0)
    n = 300
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    data["long_pnl_pct"] = rng.normal(size=n)
    data["short_pnl_pct"] = rng.normal(size=n)
    df = pd.DataFrame(data, index=index)

    result = train_classifiers(df, MLConfig(min_train_samples=200))

    assert result is not None
    assert 0.0 <= result.threshold_long <= 1.0
    assert 0.0 <= result.threshold_short <= 1.0
