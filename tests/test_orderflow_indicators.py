import numpy as np
import pandas as pd

from src.orderflow_indicators import (
    add_orderflow_risk_indicators,
    align_orderflow_to_1h,
    compute_imbalance_zscore,
    compute_oi_percentile,
)
from src.config import ORDERFLOW_STRATEGY


def test_compute_oi_percentile_flags_extremes():
    index = pd.date_range("2024-01-01", periods=21, freq="1h", tz="UTC")
    values = [1000.0] * 20 + [5000.0]
    oi = pd.Series(values, index=index)

    result = compute_oi_percentile(oi, lookback_periods=21)

    assert result.iloc[-1] == 1.0


def test_compute_imbalance_zscore_flags_extreme_buy_pressure():
    index = pd.date_range("2024-01-01", periods=21, freq="1h", tz="UTC")
    values = [0.5] * 20 + [0.95]
    ratio = pd.Series(values, index=index)

    result = compute_imbalance_zscore(ratio, lookback_periods=21)

    assert result.iloc[-1] > 1.5  # sesgo comprador claramente fuera de lo normal


def test_compute_imbalance_zscore_neutral_when_flat():
    index = pd.date_range("2024-01-01", periods=21, freq="1h", tz="UTC")
    ratio = pd.Series([0.5] * 21, index=index)

    result = compute_imbalance_zscore(ratio, lookback_periods=21)

    assert result.isna().all()  # std=0 -> sin señal, no división por cero


def test_align_orderflow_to_1h_uses_last_known_value_no_future_leak():
    oi_index = pd.date_range("2024-01-01 00:00", periods=3, freq="1h", tz="UTC")
    oi_percentile = pd.Series([0.1, 0.9, 0.2], index=oi_index)
    imbalance = pd.Series([0.0, 2.0, -2.0], index=oi_index)

    hourly_index = pd.date_range("2024-01-01 00:00", "2024-01-01 00:00", freq="1h", tz="UTC")
    df_1h = pd.DataFrame({"close": [100.0]}, index=hourly_index)

    aligned = align_orderflow_to_1h(df_1h, oi_percentile, imbalance)

    assert aligned["oi_percentile"].iloc[0] == 0.1
    assert aligned["imbalance_zscore"].iloc[0] == 0.0


def test_add_orderflow_risk_indicators_creates_expected_columns():
    rng = np.random.default_rng(0)
    n = 60
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    volume = rng.uniform(100, 1000, n)
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume}, index=index)

    result = add_orderflow_risk_indicators(df, ORDERFLOW_STRATEGY)

    assert "atr" in result.columns
    assert "volume_ma" in result.columns
