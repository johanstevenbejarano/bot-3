"""Estrategia contraria: sentimiento extremo (Fear & Greed Index) apuesta a que revierte,
confirmado por vela de rechazo y volumen real.
"""
from __future__ import annotations

import pandas as pd
from backtesting import Strategy

from src.config import SENTIMENT_STRATEGY, SentimentConfig


def compute_layers(df: pd.DataFrame, cfg: SentimentConfig = SENTIMENT_STRATEGY) -> pd.DataFrame:
    """signal_long: miedo extremo (mercado sobrevendido por pánico) + vela alcista + volumen.
    signal_short: espejo exacto sobre codicia extrema (sobrecomprado por euforia).
    """
    out = df.copy()

    out["extreme_fear"] = out["fear_greed"] <= cfg.extreme_fear_threshold
    out["extreme_greed"] = out["fear_greed"] >= cfg.extreme_greed_threshold
    out["bullish_candle"] = out["close"] > out["open"]
    out["bearish_candle"] = out["close"] < out["open"]
    out["volume_ok"] = out["volume"] > out["volume_ma"]

    out["signal_long"] = out["extreme_fear"] & out["bullish_candle"] & out["volume_ok"]
    out["signal_short"] = out["extreme_greed"] & out["bearish_candle"] & out["volume_ok"]

    return out


class SentimentContrarianStrategy(Strategy):
    """Long/short contrario a sentimiento extremo. SL/TP vía ATR, mismo techo de sizing del 20%
    que el resto de las líneas (lección de la línea 3: sin techo, un ATR chico dispara el tamaño
    de posición hasta el límite)."""

    risk_per_trade: float = SENTIMENT_STRATEGY.backtest_risk_per_trade
    sl_atr_mult: float = SENTIMENT_STRATEGY.sl_atr_mult
    tp_atr_mult: float = SENTIMENT_STRATEGY.tp_atr_mult
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
