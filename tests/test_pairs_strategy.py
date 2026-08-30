import numpy as np
import pandas as pd

from src.config import PairsConfig
from src.pairs_strategy import backtest_pairs, compute_spread_zscore


def test_proportional_series_give_near_zero_zscore():
    # B es siempre el doble de A (proporcional, no idéntico) -> el offset constante en el
    # espacio log se cancela al restar la media móvil, el z-score no debería explotar por eso.
    n = 200
    index = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    base = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    df_a = pd.DataFrame({"close": base}, index=index)
    df_b = pd.DataFrame({"close": base / 2}, index=index)

    cfg = PairsConfig(beta_window=20, z_window=20)
    result = compute_spread_zscore(df_a, df_b, cfg)

    assert not result.empty
    assert result["zscore"].abs().median() < 2.0  # sin deriva sistemática por el offset constante


def test_backtest_pairs_opens_short_a_long_b_and_closes_on_reversion():
    index = pd.date_range("2023-01-01", periods=7, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "close_a": [100, 100, 100, 100, 101, 99, 98],
            "close_b": [50, 50, 50, 50, 50, 50, 52],
            "zscore": [0.0, 0.0, 0.0, 2.5, 2.6, 1.0, 0.3],
        },
        index=index,
    )
    cfg = PairsConfig(entry_z=2.0, exit_z=0.5, stop_z=4.0)

    trades_df, stats = backtest_pairs(df, cfg, commission=0.0)

    assert len(trades_df) == 1
    trade = trades_df.iloc[0]
    assert trade["direction"] == "short_a_long_b"
    assert trade["exit_reason"] == "reverted"
    # A cayó (100->98) y B subió (50->52) durante la operación: corto en A + largo en B gana.
    assert trade["net_pnl_pct"] > 0
    assert stats.num_trades == 1
    assert stats.win_rate_pct == 100.0


def test_backtest_pairs_stops_out_on_continued_divergence():
    index = pd.date_range("2023-01-01", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "close_a": [100, 105, 110, 115, 120],
            "close_b": [50, 50, 50, 50, 50],
            "zscore": [0.0, 2.5, 3.0, 3.6, 4.0],
        },
        index=index,
    )
    cfg = PairsConfig(entry_z=2.0, exit_z=0.5, stop_z=3.5)

    trades_df, _ = backtest_pairs(df, cfg, commission=0.0)

    assert len(trades_df) == 1
    trade = trades_df.iloc[0]
    assert trade["exit_reason"] == "stopped"
    assert trade["net_pnl_pct"] < 0  # A siguió subiendo contra la posición corta


def test_no_signal_when_zscore_stays_within_entry_band():
    index = pd.date_range("2023-01-01", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "close_a": [100, 101, 99, 100, 100],
            "close_b": [50, 50, 50, 50, 50],
            "zscore": [0.2, -0.3, 0.5, -0.1, 0.0],
        },
        index=index,
    )
    cfg = PairsConfig(entry_z=2.0, exit_z=0.5, stop_z=3.5)

    trades_df, stats = backtest_pairs(df, cfg, commission=0.0)

    assert trades_df.empty
    assert stats.num_trades == 0
