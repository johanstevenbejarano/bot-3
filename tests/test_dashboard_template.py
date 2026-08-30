from datetime import datetime, timezone

from src.dashboard_render import compute_risk_levels, format_price, make_sparkline_svg
from src.dashboard_template import render_html


def _fake_symbol_data(symbol: str, price: float) -> dict:
    closes = [price - 5, price - 2, price - 3, price + 1, price]
    return {
        "symbol": symbol,
        "price": price,
        "price_fmt": format_price(price),
        "change_72h": 1.23,
        "adx": 18.4,
        "atr_pct_of_price": 2.1,
        "funding_percentile": float("nan"),
        "sparkline": make_sparkline_svg(closes),
        "risk": compute_risk_levels(price, atr=price * 0.01, sl_mult=3.0, tp_mult=6.0),
        "trend_flags": [True, False, True],
        "meanrev_flags": [False, False, True],
        "breakout_flags": [False, True],
        "signal_long": False,
        "signal_short": False,
        "active_names": [],
    }


def _fake_snapshot() -> dict:
    return {
        "generated_at": datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc),
        "symbols": {
            "BTC/USDT": _fake_symbol_data("BTC/USDT", 78000.0),
            "ETH/USDT": _fake_symbol_data("ETH/USDT", 2450.0),
        },
        "correlation": 0.72,
    }


def test_render_html_includes_per_symbol_interpretation_placeholder():
    html = render_html(_fake_snapshot())

    assert "Interpretación pendiente de esta actualización para BTC/USDT." in html
    assert "Interpretación pendiente de esta actualización para ETH/USDT." in html
    assert "lectura del presente, no una predicción" in html


def test_render_html_placeholder_is_replaceable_independently_per_symbol():
    html = render_html(_fake_snapshot())

    btc_marker = "Interpretación pendiente de esta actualización para BTC/USDT."
    eth_marker = "Interpretación pendiente de esta actualización para ETH/USDT."

    replaced = html.replace(btc_marker, "Texto de prueba BTC.")

    assert "Texto de prueba BTC." in replaced
    assert eth_marker in replaced  # el reemplazo de uno no afecta al otro
