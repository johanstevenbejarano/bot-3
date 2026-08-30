import numpy as np
import pandas as pd
import pytest

from src.seasonality_analysis import (
    analyze_by_day_of_week,
    analyze_by_group,
    analyze_by_hour,
    compute_forward_returns,
)


def test_compute_forward_returns_matches_manual_calculation():
    close = pd.Series([100.0, 110.0, 99.0, 99.0])
    result = compute_forward_returns(close, horizon_bars=1)

    assert result.iloc[0] == pytest.approx(10.0)  # 100 -> 110
    assert result.iloc[1] == pytest.approx(-10.0)  # 110 -> 99
    assert pd.isna(result.iloc[-1])  # sin vela futura


def test_compute_forward_returns_respects_horizon():
    close = pd.Series([100.0, 105.0, 110.0, 121.0])
    result = compute_forward_returns(close, horizon_bars=2)

    assert result.iloc[0] == pytest.approx(10.0)


def test_analyze_by_group_excludes_groups_below_min_samples():
    forward_returns = pd.Series([1.0] * 40 + [2.0] * 5)
    group_key = pd.Series([0] * 40 + [1] * 5)

    result = analyze_by_group(forward_returns, group_key, min_samples=30)

    assert 0 in result
    assert 1 not in result  # solo 5 muestras, por debajo del minimo


def test_analyze_by_group_detects_real_directional_bias():
    rng = np.random.default_rng(0)
    # grupo 0: retornos claramente positivos y consistentes; grupo 1: ruido puro
    group0_returns = rng.normal(loc=3.0, scale=1.0, size=200)
    group1_returns = rng.normal(loc=0.0, scale=3.0, size=200)

    forward_returns = pd.Series(np.concatenate([group0_returns, group1_returns]))
    group_key = pd.Series([0] * 200 + [1] * 200)

    result = analyze_by_group(forward_returns, group_key, min_samples=30, n_boot=1000)

    assert result[0]["excludes_zero"]
    assert result[0]["ci_low"] > 0
    assert not result[1]["excludes_zero"]


def test_analyze_by_hour_and_day_of_week_run_end_to_end():
    index = pd.date_range("2024-01-01", periods=24 * 40, freq="1h", tz="UTC")  # ~40 dias
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 1, len(index)))
    df = pd.DataFrame({"close": close}, index=index)

    by_hour = analyze_by_hour(df, min_samples=20)
    by_dow = analyze_by_day_of_week(df, min_samples=20)

    assert set(by_hour.keys()).issubset(set(range(24)))
    assert set(by_dow.keys()).issubset(set(range(7)))
    assert len(by_hour) > 0
    assert len(by_dow) > 0
