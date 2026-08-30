"""Lógica de reversión a la media (Bollinger + RSI + volumen) y wrapper para backtesting.py."""
from __future__ import annotations

import pandas as pd

from src.config import MEANREV_STRATEGY, MeanReversionConfig

try:
    from backtesting import Strategy
except ImportError:
    # No instalada en el pipeline en vivo (requirements-live.txt la excluye a propósito):
    # solo hace falta para la clase de estrategia, que ese pipeline nunca usa (solo compute_layers).
    Strategy = object


def compute_layers(df: pd.DataFrame, cfg: MeanReversionConfig = MEANREV_STRATEGY) -> pd.DataFrame:
    """Añade signal_long / signal_short: toque de banda + RSI en extremo + volumen real.

    Largo: el precio tocó/perforó la banda inferior, RSI en sobreventa, volumen por encima de su
    media — apuesta a que revierte hacia la banda central (bb_basis).
    Corto: espejo exacto en la banda superior / sobrecompra.
    """
    out = df.copy()

    out["touched_lower"] = out["low"] <= out["bb_lower"]
    out["touched_upper"] = out["high"] >= out["bb_upper"]
    out["rsi_oversold"] = out["rsi"] <= cfg.rsi.oversold
    out["rsi_overbought"] = out["rsi"] >= cfg.rsi.overbought
    out["volume_ok"] = out["volume"] > out["volume_ma"]

    out["signal_long"] = out["touched_lower"] & out["rsi_oversold"] & out["volume_ok"]
    out["signal_short"] = out["touched_upper"] & out["rsi_overbought"] & out["volume_ok"]

    return out


class MeanReversionStrategy(Strategy):
    """Long/short de reversión a la media. TP = banda central (bb_basis), SL = ATR más allá del
    extremo tocado. El sizing es riesgo fijo por operación (% de equity / distancia al SL), solo
    para poder correr el backtest — el sizing definitivo (Kelly) depende de resultados validados.
    """

    risk_per_trade: float = MEANREV_STRATEGY.risk.backtest_risk_per_trade
    sl_atr_mult: float = MEANREV_STRATEGY.risk.sl_atr_mult

    def init(self) -> None:
        self.signal_long = self.I(lambda: self.data.signal_long, name="signal_long")
        self.signal_short = self.I(lambda: self.data.signal_short, name="signal_short")
        self.atr = self.I(lambda: self.data.atr, name="atr")
        self.basis = self.I(lambda: self.data.bb_basis, name="basis")

    # Techo duro e independiente del riesgo calculado: cuando el ATR es muy pequeño (mercado en
    # calma), la distancia al SL se vuelve minúscula y el sizing por riesgo dispara el tamaño de
    # la posición hasta el límite — apostando más fuerte justo en las señales de bandas más
    # ajustadas y ruidosas. Sin este techo, un backtest con datos reales de BTC/ETH terminó en
    # -95% de drawdown por este efecto, no por la calidad real de la señal.
    max_size_fraction: float = 0.20

    def _size_fraction(self, price: float, risk_per_unit: float) -> float:
        equity_at_risk = self.equity * self.risk_per_trade
        units = equity_at_risk / risk_per_unit
        return min(self.max_size_fraction, (units * price) / self.equity)

    def next(self) -> None:
        if self.position:
            return

        price = self.data.Close[-1]
        atr = self.atr[-1]
        tp = self.basis[-1]

        if self.signal_long[-1]:
            sl = price - self.sl_atr_mult * atr
            if sl >= price or tp <= price:
                return  # la banda central ya está por debajo del precio: no hay reversión que capturar
            size_fraction = self._size_fraction(price, risk_per_unit=price - sl)
            if size_fraction > 0:
                self.buy(size=size_fraction, sl=sl, tp=tp)

        elif self.signal_short[-1]:
            sl = price + self.sl_atr_mult * atr
            if sl <= price or tp >= price:
                return
            size_fraction = self._size_fraction(price, risk_per_unit=sl - price)
            if size_fraction > 0:
                self.sell(size=size_fraction, sl=sl, tp=tp)
