import numpy as np
import pandas as pd

from src.config import STRATEGY
from src.indicators import add_indicators


def _synthetic_ohlcv(n: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.uniform(100, 1000, n)
    index = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )


def test_add_indicators_creates_expected_columns():
    df = _synthetic_ohlcv()
    result = add_indicators(df, STRATEGY)

    expected_cols = {
        "ema_fast", "ema_slow", "adx", "atr", "volume_ma", "swing_low", "swing_high", "atr_percentile",
    }
    assert expected_cols.issubset(result.columns)
    assert not result[list(expected_cols)].isna().any().any()
    assert len(result) < len(df)  # dropna() recorta el warm-up de los indicadores
