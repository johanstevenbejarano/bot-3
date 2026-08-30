import pandas as pd

from src.config import ONCHAIN_STRATEGY
from src.onchain_strategy import compute_layers


def _base_df(n: int = 3) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100] * n,
            "close": [100] * n,
            "activity_percentile": [0.5] * n,
            "volume": [100] * n,
            "volume_ma": [50] * n,
        },
        index=index,
    )


def test_long_signal_on_low_activity_extreme_with_bullish_confirmation():
    df = _base_df()
    df["activity_percentile"] = 0.05  # actividad muy baja (apatia/capitulacion)
    df["close"] = 105
    result = compute_layers(df, ONCHAIN_STRATEGY)

    assert result["activity_extreme_low"].all()
    assert result["signal_long"].all()
    assert not result["signal_short"].any()


def test_short_signal_on_high_activity_extreme_with_bearish_confirmation():
    df = _base_df()
    df["activity_percentile"] = 0.95  # actividad muy alta (euforia/pico de uso)
    df["close"] = 95
    result = compute_layers(df, ONCHAIN_STRATEGY)

    assert result["activity_extreme_high"].all()
    assert result["signal_short"].all()
    assert not result["signal_long"].any()


def test_extreme_activity_without_candle_confirmation_blocks_signal():
    df = _base_df()
    df["activity_percentile"] = 0.95
    df["close"] = 105  # no confirma un giro bajista
    result = compute_layers(df, ONCHAIN_STRATEGY)

    assert result["activity_extreme_high"].all()
    assert not result["signal_short"].any()


def test_moderate_activity_blocks_signal_even_with_confirming_candle():
    df = _base_df()
    df["activity_percentile"] = 0.5
    df["close"] = 105
    result = compute_layers(df, ONCHAIN_STRATEGY)

    assert not result["activity_extreme_high"].any()
    assert not result["activity_extreme_low"].any()
    assert not result["signal_long"].any()
    assert not result["signal_short"].any()


def test_signal_requires_volume_confirmation():
    df = _base_df()
    df["activity_percentile"] = 0.05
    df["close"] = 105
    df["volume"] = 10

    result = compute_layers(df, ONCHAIN_STRATEGY)

    assert not result["volume_ok"].any()
    assert not result["signal_long"].any()
