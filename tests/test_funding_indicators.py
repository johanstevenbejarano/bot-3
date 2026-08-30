import pandas as pd

from src.funding_indicators import align_funding_to_1h, compute_funding_percentile


def test_compute_funding_percentile_flags_extremes():
    # 20 valores estables + 1 extremo alto al final -> debería quedar en el percentil más alto
    index = pd.date_range("2024-01-01", periods=21, freq="8h", tz="UTC")
    values = [0.0001] * 20 + [0.01]
    funding = pd.Series(values, index=index)

    result = compute_funding_percentile(funding, lookback_periods=21)

    assert result.iloc[-1] == 1.0  # el valor extremo es el máximo de su propia ventana


def test_align_funding_to_1h_uses_last_known_value_no_future_leak():
    funding_index = pd.date_range("2024-01-01 00:00", periods=3, freq="8h", tz="UTC")
    funding_percentile = pd.Series([0.1, 0.9, 0.2], index=funding_index)

    # horas entre el evento de las 00:00 (0.1) y el de las 08:00 (0.9): deben ver 0.1, no 0.9
    hourly_index = pd.date_range("2024-01-01 00:00", "2024-01-01 07:00", freq="1h", tz="UTC")
    df_1h = pd.DataFrame({"close": [100] * len(hourly_index)}, index=hourly_index)

    aligned = align_funding_to_1h(df_1h, funding_percentile)

    assert aligned["funding_percentile"].eq(0.1).all()


def test_align_funding_to_1h_updates_right_after_the_next_event():
    funding_index = pd.date_range("2024-01-01 00:00", periods=3, freq="8h", tz="UTC")
    funding_percentile = pd.Series([0.1, 0.9, 0.2], index=funding_index)

    hourly_index = pd.date_range("2024-01-01 08:00", "2024-01-01 15:00", freq="1h", tz="UTC")
    df_1h = pd.DataFrame({"close": [100] * len(hourly_index)}, index=hourly_index)

    aligned = align_funding_to_1h(df_1h, funding_percentile)

    assert aligned["funding_percentile"].eq(0.9).all()
