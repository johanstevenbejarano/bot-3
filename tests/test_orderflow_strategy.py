import pandas as pd

from src.config import ORDERFLOW_STRATEGY
from src.orderflow_strategy import compute_layers


def _base_df(n: int = 3) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "oi_percentile": [0.5] * n,
            "imbalance_zscore": [0.0] * n,
            "volume": [100] * n,
            "volume_ma": [50] * n,
        },
        index=index,
    )


def test_short_signal_on_saturated_oi_with_extreme_buy_pressure():
    df = _base_df()
    df["oi_percentile"] = 0.95  # posicionamiento saturado
    df["imbalance_zscore"] = 2.0  # flujo comprador extremo -> largos sobrecargados

    result = compute_layers(df, ORDERFLOW_STRATEGY)

    assert result["oi_extreme"].all()
    assert result["imbalance_buy_extreme"].all()
    assert result["signal_short"].all()
    assert not result["signal_long"].any()


def test_long_signal_on_saturated_oi_with_extreme_sell_pressure():
    df = _base_df()
    df["oi_percentile"] = 0.95
    df["imbalance_zscore"] = -2.0  # flujo vendedor extremo -> cortos sobrecargados

    result = compute_layers(df, ORDERFLOW_STRATEGY)

    assert result["imbalance_sell_extreme"].all()
    assert result["signal_long"].all()
    assert not result["signal_short"].any()


def test_extreme_imbalance_without_saturated_oi_blocks_signal():
    df = _base_df()
    df["oi_percentile"] = 0.5  # no saturado
    df["imbalance_zscore"] = 2.0

    result = compute_layers(df, ORDERFLOW_STRATEGY)

    assert not result["oi_extreme"].any()
    assert not result["signal_short"].any()


def test_saturated_oi_without_extreme_imbalance_blocks_signal():
    df = _base_df()
    df["oi_percentile"] = 0.95
    df["imbalance_zscore"] = 0.2  # dentro de lo normal

    result = compute_layers(df, ORDERFLOW_STRATEGY)

    assert result["oi_extreme"].all()
    assert not result["signal_short"].any()
    assert not result["signal_long"].any()


def test_signal_requires_volume_confirmation():
    df = _base_df()
    df["oi_percentile"] = 0.95
    df["imbalance_zscore"] = 2.0
    df["volume"] = 10  # por debajo de su media -> sin confirmación real

    result = compute_layers(df, ORDERFLOW_STRATEGY)

    assert not result["volume_ok"].any()
    assert not result["signal_short"].any()
