import pandas as pd

from src.config import SENTIMENT_STRATEGY
from src.sentiment_strategy import compute_layers


def _base_df(n: int = 3) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100] * n,
            "close": [100] * n,
            "fear_greed": [50] * n,
            "volume": [100] * n,
            "volume_ma": [50] * n,
        },
        index=index,
    )


def test_long_signal_on_extreme_fear_with_bullish_rejection():
    df = _base_df()
    df["fear_greed"] = 10  # miedo extremo
    df["close"] = 105  # vela alcista (rebote)
    result = compute_layers(df, SENTIMENT_STRATEGY)

    assert result["extreme_fear"].all()
    assert result["signal_long"].all()
    assert not result["signal_short"].any()


def test_short_signal_on_extreme_greed_with_bearish_rejection():
    df = _base_df()
    df["fear_greed"] = 90  # codicia extrema
    df["close"] = 95  # vela bajista (rechazo)
    result = compute_layers(df, SENTIMENT_STRATEGY)

    assert result["extreme_greed"].all()
    assert result["signal_short"].all()
    assert not result["signal_long"].any()


def test_extreme_fear_without_candle_confirmation_blocks_signal():
    df = _base_df()
    df["fear_greed"] = 10
    df["close"] = 95  # vela bajista, no confirma un rebote
    result = compute_layers(df, SENTIMENT_STRATEGY)

    assert result["extreme_fear"].all()
    assert not result["signal_long"].any()


def test_moderate_sentiment_blocks_signal_even_with_confirming_candle():
    df = _base_df()
    df["fear_greed"] = 50  # ni miedo ni codicia extrema
    df["close"] = 105
    result = compute_layers(df, SENTIMENT_STRATEGY)

    assert not result["extreme_fear"].any()
    assert not result["extreme_greed"].any()
    assert not result["signal_long"].any()
    assert not result["signal_short"].any()


def test_signal_requires_volume_confirmation():
    df = _base_df()
    df["fear_greed"] = 10
    df["close"] = 105
    df["volume"] = 10  # por debajo de su media

    result = compute_layers(df, SENTIMENT_STRATEGY)

    assert not result["volume_ok"].any()
    assert not result["signal_long"].any()
