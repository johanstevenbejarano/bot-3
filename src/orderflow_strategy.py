"""Estrategia contraria de "posicionamiento saturado": interés abierto en percentil alto (mucho
apalancamiento acumulado) + flujo de órdenes fuertemente sesgado a un lado apuesta a que el
mercado revierte contra esa mayoría (short/long squeeze), confirmado por volumen real.
"""
from __future__ import annotations

import pandas as pd
from backtesting import Strategy

from src.config import ORDERFLOW_STRATEGY, OrderflowConfig


def compute_layers(df: pd.DataFrame, cfg: OrderflowConfig = ORDERFLOW_STRATEGY) -> pd.DataFrame:
    """signal_short: interés abierto saturado + flujo comprador extremo (largos sobrecargados) +
    volumen real. signal_long: espejo exacto con flujo vendedor extremo (cortos sobrecargados).
    """
    out = df.copy()

    out["oi_extreme"] = out["oi_percentile"] >= cfg.oi_extreme_percentile
    out["imbalance_buy_extreme"] = out["imbalance_zscore"] >= cfg.imbalance_extreme_z
    out["imbalance_sell_extreme"] = out["imbalance_zscore"] <= -cfg.imbalance_extreme_z
    out["volume_ok"] = out["volume"] > out["volume_ma"]

    out["signal_short"] = out["oi_extreme"] & out["imbalance_buy_extreme"] & out["volume_ok"]
    out["signal_long"] = out["oi_extreme"] & out["imbalance_sell_extreme"] & out["volume_ok"]

    return out


class OrderflowSqueezeStrategy(Strategy):
    """Long/short contrario a posicionamiento saturado. SL/TP vía ATR, mismo techo de sizing del
    20% que el resto de las líneas (lección de la línea 3: sin techo, un ATR chico dispara el
    tamaño de posición hasta el límite)."""

    risk_per_trade: float = ORDERFLOW_STRATEGY.backtest_risk_per_trade
    sl_atr_mult: float = ORDERFLOW_STRATEGY.sl_atr_mult
    tp_atr_mult: float = ORDERFLOW_STRATEGY.tp_atr_mult
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
