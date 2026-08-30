import pandas as pd

from src.onchain_indicators import align_activity_to_1h, compute_activity_percentile


def test_compute_activity_percentile_flags_extremes():
    index = pd.date_range("2024-01-01", periods=21, freq="1D", tz="UTC")
    values = [1000.0] * 20 + [5000.0]
    activity = pd.Series(values, index=index)

    result = compute_activity_percentile(activity, lookback_periods=21)

    assert result.iloc[-1] == 1.0


def test_align_activity_to_1h_uses_last_known_value_no_future_leak():
    daily_index = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    activity_percentile = pd.Series([0.1, 0.9, 0.2], index=daily_index)

    hourly_index = pd.date_range("2024-01-01 00:00", "2024-01-01 23:00", freq="1h", tz="UTC")
    df_1h = pd.DataFrame({"close": [100.0] * len(hourly_index)}, index=hourly_index)

    aligned = align_activity_to_1h(df_1h, activity_percentile)

    assert aligned["activity_percentile"].eq(0.1).all()
