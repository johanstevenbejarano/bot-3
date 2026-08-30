import pandas as pd

from src.config import MEANREV_STRATEGY
from src.meanrev_strategy import compute_layers


def _base_df(n: int = 5) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "low": [95] * n,
            "high": [105] * n,
            "close": [100] * n,
            "bb_lower": [96] * n,
            "bb_upper": [104] * n,
            "bb_basis": [100] * n,
            "rsi": [50] * n,
            "volume": [100] * n,
            "volume_ma": [50] * n,
        },
        index=index,
    )


def test_long_signal_needs_all_three_layers():
    df = _base_df()
    df["low"] = 90  # perfora la banda inferior (96)
    df["rsi"] = 25  # sobreventa
    result = compute_layers(df, MEANREV_STRATEGY)

    assert result["touched_lower"].all()
    assert result["rsi_oversold"].all()
    assert result["volume_ok"].all()
    assert result["signal_long"].all()
    assert not result["signal_short"].any()


def test_short_signal_needs_all_three_layers():
    df = _base_df()
    df["high"] = 110  # perfora la banda superior (104)
    df["rsi"] = 75  # sobrecompra
    result = compute_layers(df, MEANREV_STRATEGY)

    assert result["touched_upper"].all()
    assert result["rsi_overbought"].all()
    assert result["signal_short"].all()
    assert not result["signal_long"].any()


def test_band_touch_without_rsi_extreme_blocks_signal():
    df = _base_df()
    df["low"] = 90  # toca la banda...
    df["rsi"] = 50  # ...pero RSI no está en sobreventa
    result = compute_layers(df, MEANREV_STRATEGY)

    assert result["touched_lower"].all()
    assert not result["signal_long"].any()


def test_rsi_extreme_without_band_touch_blocks_signal():
    df = _base_df()
    df["rsi"] = 25  # sobreventa...
    df["low"] = 98  # ...pero no perfora la banda inferior (96) -> sin toque real
    result = compute_layers(df, MEANREV_STRATEGY)

    assert not result["touched_lower"].any()
    assert not result["signal_long"].any()


def test_missing_volume_blocks_signal_in_both_directions():
    df = _base_df()
    df["low"] = 90
    df["rsi"] = 25
    df["high"] = 110
    df["volume"] = 10  # por debajo de volume_ma -> rompe la confirmación
    result = compute_layers(df, MEANREV_STRATEGY)

    assert not result["volume_ok"].any()
    assert not result["signal_long"].any()
    assert not result["signal_short"].any()
