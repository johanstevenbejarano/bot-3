"""Lógica de breakout (canal de Donchian + volumen) y wrapper para backtesting.py."""
from __future__ import annotations

import pandas as pd

from src.config import BREAKOUT_STRATEGY, BreakoutConfig

try:
    from backtesting import Strategy
except ImportError:
    # No instalada en el pipeline en vivo (requirements-live.txt la excluye a propósito):
    # solo hace falta para la clase de estrategia, que ese pipeline nunca usa (solo compute_layers).
    Strategy = object


def compute_layers(df: pd.DataFrame, cfg: BreakoutConfig = BREAKOUT_STRATEGY) -> pd.DataFrame:
    """Añade signal_long / signal_short: ruptura del canal de N velas + volumen real.

    Largo: el cierre rompe por encima del máximo de las N velas previas, con volumen por encima
    de su media (descarta rupturas sin participación real). Corto: espejo exacto sobre el mínimo.
    """
    out = df.copy()

    out["breakout_up"] = out["close"] > out["donchian_high"]
    out["breakout_down"] = out["close"] < out["donchian_low"]
    out["volume_ok"] = out["volume"] > out["volume_ma"]

    out["signal_long"] = out["breakout_up"] & out["volume_ok"]
    out["signal_short"] = out["breakout_down"] & out["volume_ok"]

    # Filtro opcional de tendencia de timeframe mayor (ver src/htf_filter.py): si las columnas
    # están presentes (porque se mergearon antes de llamar a esta función), exige que la señal
    # de 1h esté a favor de la tendencia mayor. Retrocompatible: sin esas columnas, el
    # comportamiento es idéntico al de antes (líneas 4 y walk-forward ya documentados).
    if "trend_up_htf" in out.columns:
        out["signal_long"] &= out["trend_up_htf"]
    if "trend_dn_htf" in out.columns:
        out["signal_short"] &= out["trend_dn_htf"]

    return out


class BreakoutStrategy(Strategy):
    """Long/short de breakout. SL/TP simétricos vía ATR desde la entrada.

    El sizing es riesgo fijo por operación, con un techo duro independiente del riesgo calculado
    (lección de la línea 3: sin techo, un ATR pequeño dispara el tamaño de la posición hasta el
    límite justo en las señales más débiles).
    """

    risk_per_trade: float = BREAKOUT_STRATEGY.risk.backtest_risk_per_trade
    sl_atr_mult: float = BREAKOUT_STRATEGY.risk.sl_atr_mult
    tp_atr_mult: float = BREAKOUT_STRATEGY.risk.tp_atr_mult
    max_size_fraction: float = 0.20

    def init(self) -> None:
        self.signal_long = self.I(lambda: self.data.signal_long, name="signal_long")
        self.signal_short = self.I(lambda: self.data.signal_short, name="signal_short")
        self.atr = self.I(lambda: self.data.atr, name="atr")

    def _size_fraction(self, price: float, risk_per_unit: float) -> float:
        equity_at_risk = self.equity * self.risk_per_trade
        units = equity_at_risk / risk_per_unit
        return min(self.max_size_fraction, (units * price) / self.equity)

    def next(self) -> None:
        if self.position:
            return

        price = self.data.Close[-1]
        atr = self.atr[-1]

        if self.signal_long[-1]:
            sl = price - self.sl_atr_mult * atr
            tp = price + self.tp_atr_mult * atr
            if sl >= price:
                return
            size_fraction = self._size_fraction(price, risk_per_unit=price - sl)
            if size_fraction > 0:
                self.buy(size=size_fraction, sl=sl, tp=tp)

        elif self.signal_short[-1]:
            sl = price + self.sl_atr_mult * atr
            tp = price - self.tp_atr_mult * atr
            if sl <= price or tp <= 0:
                return
            size_fraction = self._size_fraction(price, risk_per_unit=sl - price)
            if size_fraction > 0:
                self.sell(size=size_fraction, sl=sl, tp=tp)
