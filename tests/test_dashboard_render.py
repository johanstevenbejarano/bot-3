import pytest

from src.dashboard_render import (
    compute_risk_levels,
    format_price,
    layers_met_fraction,
    make_sparkline_svg,
    pct_change,
)


def test_sparkline_endpoint_matches_last_value_extremes():
    # serie estrictamente creciente -> el último punto debe quedar en la parte más alta (y mínima)
    values = [1, 2, 3, 4, 5]
    result = make_sparkline_svg(values, width=100, height=50, pad=5)

    assert result["endpoint_y"] == pytest.approx(5.0, abs=0.1)  # arriba del todo (y chico = arriba)
    assert result["endpoint_x"] == pytest.approx(95.0, abs=0.1)  # extremo derecho


def test_sparkline_flat_series_does_not_crash():
    result = make_sparkline_svg([100, 100, 100, 100])
    assert result["line_points"]  # no lanza ZeroDivisionError con span=0


def test_sparkline_requires_at_least_two_points():
    with pytest.raises(ValueError):
        make_sparkline_svg([42])


def test_pct_change_up_and_down():
    assert pct_change([100, 110]) == pytest.approx(10.0)
    assert pct_change([100, 90]) == pytest.approx(-10.0)


def test_pct_change_handles_degenerate_input():
    assert pct_change([100]) == 0.0
    assert pct_change([]) == 0.0


def test_format_price_uses_thousands_separator_and_two_decimals():
    assert format_price(78019.8) == "$78,019.80"
    assert format_price(2456.08) == "$2,456.08"


def test_format_price_uses_more_decimals_for_small_values():
    assert format_price(0.0342) == "$0.034200"


def test_compute_risk_levels_symmetric_around_price():
    levels = compute_risk_levels(price=100, atr=2, sl_mult=3, tp_mult=8)

    assert levels.long_sl == 94
    assert levels.long_tp == 116
    assert levels.short_sl == 106
    assert levels.short_tp == 84


def test_layers_met_fraction():
    assert layers_met_fraction([True, False, True]) == "2/3"
    assert layers_met_fraction([False, False]) == "0/2"
    assert layers_met_fraction([True, True, True]) == "3/3"
