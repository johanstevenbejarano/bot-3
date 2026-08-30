import pandas as pd

from src.sentiment_indicators import align_sentiment_to_1h


def test_align_sentiment_to_1h_uses_last_known_value_no_future_leak():
    daily_index = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    fear_greed = pd.Series([10.0, 90.0, 50.0], index=daily_index)

    # horas entre el evento del dia 1 (10) y el del dia 2 (90): deben ver 10, no 90
    hourly_index = pd.date_range("2024-01-01 00:00", "2024-01-01 23:00", freq="1h", tz="UTC")
    df_1h = pd.DataFrame({"close": [100.0] * len(hourly_index)}, index=hourly_index)

    aligned = align_sentiment_to_1h(df_1h, fear_greed)

    assert aligned["fear_greed"].eq(10.0).all()


def test_align_sentiment_to_1h_updates_right_after_the_next_event():
    daily_index = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    fear_greed = pd.Series([10.0, 90.0, 50.0], index=daily_index)

    hourly_index = pd.date_range("2024-01-02 00:00", "2024-01-02 23:00", freq="1h", tz="UTC")
    df_1h = pd.DataFrame({"close": [100.0] * len(hourly_index)}, index=hourly_index)

    aligned = align_sentiment_to_1h(df_1h, fear_greed)

    assert aligned["fear_greed"].eq(90.0).all()
