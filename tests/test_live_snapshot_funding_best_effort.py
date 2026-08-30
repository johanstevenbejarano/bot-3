import pandas as pd
import pytest

from src.funding_indicators import align_funding_to_1h, compute_funding_percentile
from src.live_snapshot import _fetch_funding_best_effort


def test_returns_empty_series_when_fetch_raises(monkeypatch):
    def _boom(symbol: str) -> pd.Series:
        raise RuntimeError("451 restricted location")

    monkeypatch.setattr("src.live_snapshot.fetch_recent_funding", _boom)

    result = _fetch_funding_best_effort("BTC/USDT")

    assert isinstance(result, pd.Series)
    assert result.empty
    assert isinstance(result.index, pd.DatetimeIndex)


def test_returns_actual_series_when_fetch_succeeds(monkeypatch):
    expected = pd.Series([0.0001, 0.0002], index=pd.date_range("2026-01-01", periods=2, freq="8h", tz="UTC"))

    monkeypatch.setattr("src.live_snapshot.fetch_recent_funding", lambda symbol: expected)

    result = _fetch_funding_best_effort("BTC/USDT")

    pd.testing.assert_series_equal(result, expected)


def test_empty_funding_series_flows_to_all_nan_percentile_without_raising():
    empty_funding = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    index_1h = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
    df_1h = pd.DataFrame({"close": range(10)}, index=index_1h)

    funding_percentile = compute_funding_percentile(empty_funding, lookback_periods=270)
    aligned = align_funding_to_1h(df_1h, funding_percentile)

    assert aligned["funding_percentile"].isna().all()
