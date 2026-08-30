import pandas as pd

from src.macro_indicators import align_dxy_to_1h, compute_dxy_percentile


def test_compute_dxy_percentile_flags_extremes():
    index = pd.date_range("2024-01-01", periods=21, freq="1D", tz="UTC")
    values = [100.0] * 20 + [110.0]
    dxy = pd.Series(values, index=index)

    result = compute_dxy_percentile(dxy, lookback_periods=21)

    assert result.iloc[-1] == 1.0


def test_align_dxy_to_1h_uses_last_known_value_no_future_leak():
    daily_index = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    dxy_percentile = pd.Series([0.1, 0.9, 0.2], index=daily_index)

    hourly_index = pd.date_range("2024-01-01 00:00", "2024-01-01 23:00", freq="1h", tz="UTC")
    df_1h = pd.DataFrame({"close": [100.0] * len(hourly_index)}, index=hourly_index)

    aligned = align_dxy_to_1h(df_1h, dxy_percentile)

    assert aligned["dxy_percentile"].eq(0.1).all()


def test_align_dxy_to_1h_holds_last_value_through_a_weekend_gap():
    # el DXY no cotiza fin de semana -- solo hay eventos viernes y lunes, pero cripto sigue 24/7
    daily_index = pd.to_datetime(["2024-01-05", "2024-01-08"], utc=True)  # viernes, lunes
    dxy_percentile = pd.Series([0.3, 0.7], index=daily_index)

    hourly_index = pd.date_range("2024-01-06 00:00", "2024-01-07 23:00", freq="1h", tz="UTC")  # sabado y domingo
    df_1h = pd.DataFrame({"close": [100.0] * len(hourly_index)}, index=hourly_index)

    aligned = align_dxy_to_1h(df_1h, dxy_percentile)

    assert aligned["dxy_percentile"].eq(0.3).all()  # mantiene el valor del viernes todo el fin de semana
