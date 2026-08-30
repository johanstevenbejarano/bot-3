import numpy as np
import pandas as pd
import pytest

from src.ta_free_indicators import adx, atr, bollinger_bands, ema, rsi, true_range


def _synthetic_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.normal(0, 0.3, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_ema_converges_to_constant_series():
    s = pd.Series([50.0] * 100)
    result = ema(s, window=10)
    assert result.iloc[-1] == pytest.approx(50.0)


def test_ema_reacts_faster_with_smaller_window():
    s = pd.Series([1.0] * 20 + [10.0] * 20)
    fast = ema(s, window=3)
    slow = ema(s, window=20)
    assert fast.iloc[21] > slow.iloc[21]


def test_true_range_non_negative_and_at_least_high_low_range():
    df = _synthetic_ohlcv()
    tr = true_range(df["high"], df["low"], df["close"])
    valid = tr.dropna()
    assert (valid >= 0).all()
    assert (valid.iloc[1:] >= (df["high"] - df["low"]).iloc[1:] - 1e-9).all()


def test_atr_non_negative_and_has_warmup_nans():
    df = _synthetic_ohlcv()
    result = atr(df["high"], df["low"], df["close"], window=14)
    assert result.iloc[:13].isna().all()
    assert (result.dropna() >= 0).all()


def test_rsi_bounds_and_extremes():
    df = _synthetic_ohlcv()
    result = rsi(df["close"], window=14).dropna()
    assert (result >= 0).all() and (result <= 100).all()

    rising = pd.Series(np.arange(1, 60, dtype=float))
    rsi_rising = rsi(rising, window=14).dropna()
    assert rsi_rising.iloc[-1] > 95

    falling = pd.Series(np.arange(60, 1, -1, dtype=float))
    rsi_falling = rsi(falling, window=14).dropna()
    assert rsi_falling.iloc[-1] < 5


def test_adx_non_negative():
    df = _synthetic_ohlcv()
    result = adx(df["high"], df["low"], df["close"], window=14).dropna()
    assert (result >= 0).all()
    assert (result <= 100).all()


def test_bollinger_bands_ordering_and_basis_is_sma():
    df = _synthetic_ohlcv()
    basis, upper, lower = bollinger_bands(df["close"], window=20, num_std=2.0)
    valid = basis.notna()
    assert (upper[valid] >= basis[valid]).all()
    assert (basis[valid] >= lower[valid]).all()
    expected_basis = df["close"].rolling(20).mean()
    pd.testing.assert_series_equal(basis, expected_basis, check_names=False)
