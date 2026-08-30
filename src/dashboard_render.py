"""Funciones puras (sin red, sin estado) para calcular y formatear lo que el dashboard muestra.
Separadas de `live_snapshot.py` (que sí hace llamadas de red) para poder testearlas sin mockear
nada.
"""
from __future__ import annotations

from dataclasses import dataclass


def make_sparkline_svg(
    values: list[float], width: float = 280, height: float = 56, pad: float = 4
) -> dict:
    """Coordenadas de un sparkline (polyline + área + punto final) a partir de una serie de
    precios. No depende del navegador: son números ya calculados, listos para insertar en SVG.
    """
    if len(values) < 2:
        raise ValueError("se necesitan al menos 2 valores para un sparkline")

    lo, hi = min(values), max(values)
    span = hi - lo if hi != lo else 1
    n = len(values)

    points = []
    for i, v in enumerate(values):
        x = pad + i * (width - 2 * pad) / (n - 1)
        y = height - pad - (v - lo) / span * (height - 2 * pad)
        points.append((round(x, 1), round(y, 1)))

    line = " ".join(f"{x},{y}" for x, y in points)
    area = f"{pad},{height - pad} " + line + f" {width - pad},{height - pad}"

    return {
        "line_points": line,
        "area_points": area,
        "endpoint_x": points[-1][0],
        "endpoint_y": points[-1][1],
    }


def pct_change(values: list[float]) -> float:
    """% de cambio entre el primer y último valor de la serie."""
    if len(values) < 2 or values[0] == 0:
        return 0.0
    return (values[-1] / values[0] - 1) * 100


def format_price(value: float) -> str:
    """`78019.8` -> `$78,019.80`. Usa 2 decimales para precios >= 1, más para precios chicos
    (ej. tokens de pocos centavos) para no perder precisión visible."""
    decimals = 2 if value >= 1 else 6
    return f"${value:,.{decimals}f}"


@dataclass(frozen=True)
class RiskLevels:
    long_sl: float
    long_tp: float
    short_sl: float
    short_tp: float


def compute_risk_levels(price: float, atr: float, sl_mult: float, tp_mult: float) -> RiskLevels:
    """Niveles de SL/TP de referencia vía ATR, simétricos para largo y corto — misma fórmula
    usada en las estrategias de la sesión (breakout_strategy.py, meanrev_strategy.py, etc.)."""
    return RiskLevels(
        long_sl=price - sl_mult * atr,
        long_tp=price + tp_mult * atr,
        short_sl=price + sl_mult * atr,
        short_tp=price - tp_mult * atr,
    )


def layers_met_fraction(flags: list[bool]) -> str:
    """[True, False, True] -> '2/3'."""
    return f"{sum(1 for f in flags if f)}/{len(flags)}"
