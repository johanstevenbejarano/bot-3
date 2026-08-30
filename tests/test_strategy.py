import pandas as pd

from src.config import STRATEGY
from src.strategy import compute_layers


def _long_setup_df(n: int = 6) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [103, 100, 99, 100, 104, 105],
            "close": [104, 99.5, 98, 103, 106, 107],
            "high": [106, 101, 100, 106, 107, 108],
            "low": [105, 99, 99, 105, 105, 105],
            "ema_fast": [100] * n,
            "ema_slow": [90] * n,  # ema_fast > ema_slow -> tendencia alcista
            "adx": [30] * n,
            "volume": [100] * n,
            "volume_ma": [50] * n,
            "atr": [1] * n,
            "swing_low": [95] * n,
            "swing_high": [110] * n,
            "atr_percentile": [0.3] * n,  # régimen de volatilidad normal -> no bloquea
        },
        index=index,
    )


def _short_setup_df(n: int = 6) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [97, 100, 101, 100, 96, 95],
            "close": [96, 100.5, 102, 97, 94, 93],
            "high": [95, 101, 101, 95, 95, 95],
            "low": [94, 99, 100, 94, 93, 92],
            "ema_fast": [100] * n,
            "ema_slow": [110] * n,  # ema_fast < ema_slow -> tendencia bajista
            "adx": [30] * n,
            "volume": [100] * n,
            "volume_ma": [50] * n,
            "atr": [1] * n,
            "swing_low": [90] * n,
            "swing_high": [105] * n,
            "atr_percentile": [0.3] * n,  # régimen de volatilidad normal -> no bloquea
        },
        index=index,
    )


def test_long_pullback_and_reclaim_triggers_signal():
    df = _long_setup_df()
    result = compute_layers(df, STRATEGY)

    assert not result.iloc[1]["pullback_up_ok"]
    assert not result.iloc[2]["pullback_up_ok"]
    assert result.iloc[3]["pullback_up_ok"]
    assert result.iloc[3]["signal_long"]
    assert not result["signal_short"].any()


def test_short_rally_and_rejection_triggers_signal():
    df = _short_setup_df()
    result = compute_layers(df, STRATEGY)

    assert not result.iloc[1]["pullback_dn_ok"]
    assert not result.iloc[2]["pullback_dn_ok"]
    assert result.iloc[3]["pullback_dn_ok"]
    assert result.iloc[3]["signal_short"]
    assert not result["signal_long"].any()


def test_no_recent_touch_blocks_long_signal():
    df = _long_setup_df()
    df["low"] = [105] * len(df)  # el precio nunca retrocede hasta la EMA rápida
    result = compute_layers(df, STRATEGY)

    assert not result["pullback_up_ok"].any()
    assert not result["signal_long"].any()


def test_missing_trend_layer_blocks_long_signal():
    df = _long_setup_df()
    df["ema_fast"] = 80  # ema_fast < ema_slow -> rompe la capa de tendencia alcista
    result = compute_layers(df, STRATEGY)

    assert not result["trend_up_ok"].any()
    assert not result["signal_long"].any()


def test_missing_volume_layer_blocks_signal_in_both_directions():
    df = _long_setup_df()
    df["volume"] = 10  # por debajo de volume_ma -> rompe la capa de volumen
    result = compute_layers(df, STRATEGY)

    assert not result["volume_ok"].any()
    assert not result["signal_long"].any()
    assert not result["signal_short"].any()


def test_high_volatility_regime_blocks_signal_even_with_all_layers_met():
    df = _long_setup_df()
    df["atr_percentile"] = 0.9  # por encima del umbral -> régimen de alta volatilidad
    result = compute_layers(df, STRATEGY)

    assert not result["regime_ok"].any()
    assert result.iloc[3]["pullback_up_ok"]  # las 3 capas se siguen cumpliendo...
    assert not result.iloc[3]["signal_long"]  # ...pero el régimen bloquea la señal
