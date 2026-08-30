import pandas as pd

from src.config import FUNDING_STRATEGY
from src.funding_strategy import compute_layers


def _base_df(n: int = 3) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100] * n,
            "close": [100] * n,
            "funding_percentile": [0.5] * n,
            "volume": [100] * n,
            "volume_ma": [50] * n,
        },
        index=index,
    )


def test_short_signal_on_extreme_high_funding_with_bearish_rejection():
    df = _base_df()
    df["funding_percentile"] = 0.95  # extremo alto (largos sobrecargados)
    df["close"] = 95  # vela bajista (close < open)
    result = compute_layers(df, FUNDING_STRATEGY)

    assert result["funding_extreme_high"].all()
    assert result["signal_short"].all()
    assert not result["signal_long"].any()


def test_long_signal_on_extreme_low_funding_with_bullish_rejection():
    df = _base_df()
    df["funding_percentile"] = 0.05  # extremo bajo (cortos sobrecargados)
    df["close"] = 105  # vela alcista
    result = compute_layers(df, FUNDING_STRATEGY)

    assert result["funding_extreme_low"].all()
    assert result["signal_long"].all()
    assert not result["signal_short"].any()


def test_extreme_funding_without_candle_confirmation_blocks_signal():
    df = _base_df()
    df["funding_percentile"] = 0.95
    df["close"] = 105  # vela alcista, no confirma un giro bajista
    result = compute_layers(df, FUNDING_STRATEGY)

    assert result["funding_extreme_high"].all()
    assert not result["signal_short"].any()


def test_moderate_funding_blocks_signal_even_with_confirming_candle():
    df = _base_df()
    df["funding_percentile"] = 0.5  # no extremo
    df["close"] = 95
    result = compute_layers(df, FUNDING_STRATEGY)

    assert not result["funding_extreme_high"].any()
    assert not result["signal_short"].any()
