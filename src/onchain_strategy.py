"""Estrategia contraria: actividad on-chain (direcciones activas) en un extremo de su propia
historia reciente apuesta a que el precio revierte, confirmado por vela de rechazo y volumen.
"""
from __future__ import annotations

import pandas as pd
from backtesting import Strategy

from src.config import ONCHAIN_STRATEGY, OnchainConfig


def compute_layers(df: pd.DataFrame, cfg: OnchainConfig = ONCHAIN_STRATEGY) -> pd.DataFrame:
    """signal_short: actividad en percentil alto extremo (euforia/uso pico) + vela bajista +
    volumen. signal_long: espejo exacto sobre percentil bajo extremo (apatía/capitulación)."""
    out = df.copy()

    out["activity_extreme_high"] = out["activity_percentile"] >= cfg.extreme_percentile
    out["activity_extreme_low"] = out["activity_percentile"] <= (1 - cfg.extreme_percentile)
    out["bullish_candle"] = out["close"] > out["open"]
    out["bearish_candle"] = out["close"] < out["open"]
    out["volume_ok"] = out["volume"] > out["volume_ma"]

    out["signal_long"] = out["activity_extreme_low"] & out["bullish_candle"] & out["volume_ok"]
    out["signal_short"] = out["activity_extreme_high"] & out["bearish_candle"] & out["volume_ok"]

    return out


class OnchainContrarianStrategy(Strategy):
    """Long/short contrario a un extremo de actividad on-chain. SL/TP vía ATR, mismo techo de
    sizing del 20% que el resto de las líneas."""

    risk_per_trade: float = ONCHAIN_STRATEGY.backtest_risk_per_trade
    sl_atr_mult: float = ONCHAIN_STRATEGY.sl_atr_mult
    tp_atr_mult: float = ONCHAIN_STRATEGY.tp_atr_mult
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
