import pandas as pd

from src.config import BREAKOUT_STRATEGY
from src.breakout_strategy import compute_layers


def _base_df(n: int = 3) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "close": [100] * n,
            "donchian_high": [105] * n,
            "donchian_low": [95] * n,
            "volume": [100] * n,
            "volume_ma": [50] * n,
        },
        index=index,
    )


def test_long_signal_on_upside_breakout_with_volume():
    df = _base_df()
    df["close"] = 110  # rompe por encima del canal (105)
    result = compute_layers(df, BREAKOUT_STRATEGY)

    assert result["breakout_up"].all()
    assert result["signal_long"].all()
    assert not result["signal_short"].any()


def test_short_signal_on_downside_breakout_with_volume():
    df = _base_df()
    df["close"] = 90  # rompe por debajo del canal (95)
    result = compute_layers(df, BREAKOUT_STRATEGY)

    assert result["breakout_down"].all()
    assert result["signal_short"].all()
    assert not result["signal_long"].any()


def test_breakout_without_volume_confirmation_blocks_signal():
    df = _base_df()
    df["close"] = 110
    df["volume"] = 10  # por debajo de volume_ma -> sin confirmación real
    result = compute_layers(df, BREAKOUT_STRATEGY)

    assert result["breakout_up"].all()
    assert not result["volume_ok"].any()
    assert not result["signal_long"].any()


def test_no_breakout_inside_channel_blocks_signal():
    df = _base_df()  # close=100, dentro del canal [95, 105]
    result = compute_layers(df, BREAKOUT_STRATEGY)

    assert not result["breakout_up"].any()
    assert not result["breakout_down"].any()
    assert not result["signal_long"].any()
    assert not result["signal_short"].any()


def test_htf_filter_blocks_breakout_against_the_higher_timeframe_trend():
    df = _base_df()
    df["close"] = 110  # ruptura alcista limpia con volumen...
    df["trend_up_htf"] = False  # ...pero la tendencia diaria es bajista
    df["trend_dn_htf"] = True
    result = compute_layers(df, BREAKOUT_STRATEGY)

    assert result["breakout_up"].all()  # la ruptura de 1h sigue ahí...
    assert not result["signal_long"].any()  # ...pero el filtro HTF la bloquea


def test_htf_filter_allows_breakout_aligned_with_higher_timeframe_trend():
    df = _base_df()
    df["close"] = 110
    df["trend_up_htf"] = True  # tendencia diaria a favor
    df["trend_dn_htf"] = False
    result = compute_layers(df, BREAKOUT_STRATEGY)

    assert result["signal_long"].all()


def test_without_htf_columns_behavior_is_unchanged():
    df = _base_df()
    df["close"] = 110
    assert "trend_up_htf" not in df.columns
    result = compute_layers(df, BREAKOUT_STRATEGY)

    assert result["signal_long"].all()  # sin columnas HTF, se comporta como antes (línea 4)
