import pandas as pd
import pytest

from src.funding_arb_strategy import backtest_funding_arb, compute_episode_return_pct, find_episodes


def test_find_episodes_basic_entry_and_exit():
    index = pd.date_range("2024-01-01", periods=6, freq="8h", tz="UTC")
    funding_rate = pd.Series([0.0001, 0.0005, 0.0006, 0.0001, -0.0002, 0.0003], index=index)

    episodes = find_episodes(funding_rate, entry_threshold=0.0002, exit_threshold=0.0)

    # entra en t=1 (0.0005 > 0.0002), sale en t=4 (-0.0002 <= 0)
    assert episodes == [(index[1], index[4])]


def test_find_episodes_multiple_windows():
    index = pd.date_range("2024-01-01", periods=8, freq="8h", tz="UTC")
    funding_rate = pd.Series([0.001, -0.001, 0.001, -0.001, 0.001, -0.001, 0.001, -0.001], index=index)

    episodes = find_episodes(funding_rate, entry_threshold=0.0, exit_threshold=0.0)

    assert episodes == [
        (index[0], index[1]),
        (index[2], index[3]),
        (index[4], index[5]),
        (index[6], index[7]),
    ]


def test_find_episodes_discards_unclosed_final_window():
    index = pd.date_range("2024-01-01", periods=3, freq="8h", tz="UTC")
    funding_rate = pd.Series([0.001, 0.001, 0.001], index=index)  # nunca cae por debajo del umbral

    episodes = find_episodes(funding_rate, entry_threshold=0.0, exit_threshold=-1.0)

    assert episodes == []  # sigue "adentro" al final -> no hay precio de salida real, se descarta


def test_compute_episode_return_pct_pure_funding_no_basis_change():
    index = pd.date_range("2024-01-01", periods=3, freq="8h", tz="UTC")
    spot = pd.Series([100.0, 100.0, 100.0], index=index)
    perp = pd.Series([100.5, 100.5, 100.5], index=index)  # base constante -> sin riesgo de precio
    funding_rate = pd.Series([0.001, 0.001, 0.001], index=index)  # 0.1% cada 8h

    result = compute_episode_return_pct(index[0], index[2], spot, perp, funding_rate, commission=0.0)

    # funding cobrado: solo el evento en index[0] cae dentro de [entry, exit) -- index[1] tambien, index[2] es la salida (excluido)
    assert result == pytest.approx(0.2)  # 2 eventos de 0.001 = 0.2%


def test_compute_episode_return_pct_widening_basis_hurts_pnl():
    index = pd.date_range("2024-01-01", periods=2, freq="8h", tz="UTC")
    spot = pd.Series([100.0, 100.0], index=index)
    perp = pd.Series([100.0, 101.0], index=index)  # el perpetuo sube respecto al spot -> la base se ensancha
    funding_rate = pd.Series([0.0, 0.0], index=index)  # sin funding, solo riesgo de precio

    result = compute_episode_return_pct(index[0], index[1], spot, perp, funding_rate, commission=0.0)

    # entry_basis=0, exit_basis=1 -> price_pnl = (0 - 1) / 100 * 100 = -1%
    assert result == pytest.approx(-1.0)


def test_compute_episode_return_pct_subtracts_round_trip_commission():
    index = pd.date_range("2024-01-01", periods=2, freq="8h", tz="UTC")
    spot = pd.Series([100.0, 100.0], index=index)
    perp = pd.Series([100.0, 100.0], index=index)
    funding_rate = pd.Series([0.0, 0.0], index=index)

    result = compute_episode_return_pct(index[0], index[1], spot, perp, funding_rate, commission=0.001)

    assert result == pytest.approx(-0.4)  # 4 * 0.001 * 100 = 0.4% de costo, sin nada que lo compense


def test_backtest_funding_arb_discards_episodes_before_price_history_starts():
    # el funding empieza antes que el precio del perpetuo (caso real: BTC perp arranca 7h
    # despues del primer evento de funding) -- ese episodio no deberia contaminar el promedio.
    funding_index = pd.date_range("2024-01-01 00:00", periods=4, freq="8h", tz="UTC")
    funding_rate = pd.Series([0.001, -0.001, 0.001, -0.001], index=funding_index)

    price_index = pd.date_range("2024-01-01 08:00", periods=3, freq="8h", tz="UTC")  # arranca despues del primer evento
    spot = pd.Series([100.0, 100.0, 100.0], index=price_index)
    perp = pd.Series([100.0, 100.0, 100.0], index=price_index)

    results = backtest_funding_arb(spot, perp, funding_rate, entry_threshold=0.0, exit_threshold=0.0, commission=0.0)

    # primer episodio (entra 00:00, sale 08:00) tiene entrada sin precio -> se descarta, no genera NaN
    assert len(results) == 1
    assert all(pd.notna(r) for r in results)


def test_backtest_funding_arb_returns_one_value_per_episode():
    index = pd.date_range("2024-01-01", periods=4, freq="8h", tz="UTC")
    spot = pd.Series([100.0] * 4, index=index)
    perp = pd.Series([100.0] * 4, index=index)
    funding_rate = pd.Series([0.001, 0.001, -0.001, 0.001], index=index)

    results = backtest_funding_arb(spot, perp, funding_rate, entry_threshold=0.0, exit_threshold=0.0, commission=0.0)

    assert len(results) == 1  # un solo episodio cerrado (entra en 0, sale en 2); el ultimo evento queda sin cerrar
