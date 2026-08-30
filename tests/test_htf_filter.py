import numpy as np
import pandas as pd

from src.htf_filter import _shift_and_align, compute_htf_trend


def test_shift_and_align_uses_only_previous_days_confirmed_trend():
    daily_index = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")
    daily_bool = pd.DataFrame({"trend_up_htf": [True, True, False, False]}, index=daily_index)

    # Horas dentro del 2024-01-03: la tendencia "de ese día" ya es False, pero la CONFIRMADA
    # (la del día anterior, 01-02) todavía era True.
    hourly_index = pd.date_range("2024-01-03 00:00", "2024-01-03 23:00", freq="1h", tz="UTC")

    aligned = _shift_and_align(daily_bool, hourly_index)

    assert aligned["trend_up_htf"].eq(True).all()


def test_shift_and_align_updates_the_day_after_the_flip():
    daily_index = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")
    daily_bool = pd.DataFrame({"trend_up_htf": [True, True, False, False]}, index=daily_index)

    # 2024-01-04: el día anterior (01-03) ya cerró en False -> ahora sí debe reflejarse.
    hourly_index = pd.date_range("2024-01-04 00:00", "2024-01-04 23:00", freq="1h", tz="UTC")

    aligned = _shift_and_align(daily_bool, hourly_index)

    assert aligned["trend_up_htf"].eq(False).all()


def test_compute_htf_trend_flags_close_above_and_below_ema():
    index = pd.date_range("2024-01-01", periods=100, freq="1D", tz="UTC")
    close = pd.Series(np.concatenate([np.full(50, 100.0), np.full(50, 200.0)]), index=index)

    result = compute_htf_trend(close, ema_period=10)

    # tras suficiente historia, la EMA converge y el segundo tramo (precio mucho más alto) queda
    # claramente por encima de su propia EMA.
    assert result["trend_up_htf"].iloc[-1]
    assert not result["trend_dn_htf"].iloc[-1]
    # trend_up_htf y trend_dn_htf son mutuamente excluyentes
    assert not (result["trend_up_htf"] & result["trend_dn_htf"]).any()
