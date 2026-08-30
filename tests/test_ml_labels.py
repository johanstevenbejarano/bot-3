import pandas as pd

from src.ml_labels import triple_barrier_labels


def _df(rows: list[dict]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(rows), freq="1h", tz="UTC")
    return pd.DataFrame(rows, index=index)


def test_long_hits_tp_before_sl():
    # entry=100, atr=1, sl_mult=2 -> sl=98, tp_mult=3 -> tp=103
    rows = [
        {"close": 100, "high": 100, "low": 100, "atr": 1},
        {"close": 100, "high": 101, "low": 99, "atr": 1},   # ni tp ni sl
        {"close": 104, "high": 104, "low": 100, "atr": 1},  # toca tp (103)
        {"close": 100, "high": 100, "low": 100, "atr": 1},
    ]
    df = _df(rows)
    result = triple_barrier_labels(df, sl_atr_mult=2.0, tp_atr_mult=3.0, max_holding_bars=3)

    assert result.iloc[0]["long_exit_offset"] == 2
    assert result.iloc[0]["long_pnl_pct"] > 0
    assert abs(result.iloc[0]["long_pnl_pct"] - 3.0) < 1e-9  # (103/100 - 1) * 100


def test_long_hits_sl_before_tp():
    rows = [
        {"close": 100, "high": 100, "low": 100, "atr": 1},
        {"close": 97, "high": 100, "low": 97, "atr": 1},  # toca sl (98)
        {"close": 100, "high": 100, "low": 100, "atr": 1},
        {"close": 100, "high": 100, "low": 100, "atr": 1},
    ]
    df = _df(rows)
    result = triple_barrier_labels(df, sl_atr_mult=2.0, tp_atr_mult=3.0, max_holding_bars=3)

    assert result.iloc[0]["long_exit_offset"] == 1
    assert result.iloc[0]["long_pnl_pct"] < 0
    assert abs(result.iloc[0]["long_pnl_pct"] - (-2.0)) < 1e-9  # (98/100 - 1) * 100


def test_short_hits_tp_before_sl():
    # entry=100, sl_short=102, tp_short=97
    rows = [
        {"close": 100, "high": 100, "low": 100, "atr": 1},
        {"close": 96, "high": 100, "low": 96, "atr": 1},  # toca tp_short (97)
        {"close": 100, "high": 100, "low": 100, "atr": 1},
        {"close": 100, "high": 100, "low": 100, "atr": 1},
    ]
    df = _df(rows)
    result = triple_barrier_labels(df, sl_atr_mult=2.0, tp_atr_mult=3.0, max_holding_bars=3)

    assert result.iloc[0]["short_exit_offset"] == 1
    assert result.iloc[0]["short_pnl_pct"] > 0
    assert abs(result.iloc[0]["short_pnl_pct"] - 3.0) < 1e-9  # (100-97)/100*100


def test_both_barriers_touched_same_bar_assumes_sl_first():
    rows = [
        {"close": 100, "high": 100, "low": 100, "atr": 1},
        {"close": 100, "high": 105, "low": 95, "atr": 1},  # toca sl (98) Y tp (103) en la misma vela
        {"close": 100, "high": 100, "low": 100, "atr": 1},
        {"close": 100, "high": 100, "low": 100, "atr": 1},
    ]
    df = _df(rows)
    result = triple_barrier_labels(df, sl_atr_mult=2.0, tp_atr_mult=3.0, max_holding_bars=3)

    assert result.iloc[0]["long_pnl_pct"] < 0  # asume SL primero (conservador), no TP


def test_neither_barrier_touched_uses_time_barrier_close():
    rows = [
        {"close": 100, "high": 100, "low": 100, "atr": 1},
        {"close": 100.5, "high": 100.5, "low": 99.5, "atr": 1},
        {"close": 100.8, "high": 100.8, "low": 99.5, "atr": 1},
        {"close": 100.5, "high": 100.5, "low": 99.5, "atr": 1},  # barrera de tiempo (hold=3)
    ]
    df = _df(rows)
    result = triple_barrier_labels(df, sl_atr_mult=2.0, tp_atr_mult=3.0, max_holding_bars=3)

    assert result.iloc[0]["long_exit_offset"] == 3
    assert abs(result.iloc[0]["long_pnl_pct"] - 0.5) < 1e-9  # (100.5/100 - 1) * 100


def test_last_rows_without_enough_future_data_are_excluded():
    rows = [{"close": 100, "high": 100, "low": 100, "atr": 1} for _ in range(5)]
    df = _df(rows)
    result = triple_barrier_labels(df, sl_atr_mult=2.0, tp_atr_mult=3.0, max_holding_bars=3)

    # con max_holding_bars=3 y 5 filas, solo la fila 0 y 1 tienen suficiente futuro (5-3-1=1 -> 2 filas)
    assert len(result) == 2
