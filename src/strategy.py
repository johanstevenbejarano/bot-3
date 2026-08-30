"""Lógica de confluencia (tendencia + pullback + volumen), largo y corto, y wrapper para
backtesting.py.
"""
from __future__ import annotations

import pandas as pd

from src.config import STRATEGY, StrategyConfig

try:
    from backtesting import Strategy
except ImportError:
    # No instalada en el pipeline en vivo (requirements-live.txt la excluye a propósito):
    # solo hace falta para ConfluenceStrategy, que ese pipeline nunca usa (solo compute_layers).
    Strategy = object


def _rolling_any(cond: pd.Series, window: int) -> pd.Series:
    return cond.rolling(window=window, min_periods=1).max().astype(bool)


def compute_layers(df: pd.DataFrame, cfg: StrategyConfig = STRATEGY) -> pd.DataFrame:
    """Añade las 3 capas por dirección y signal_long / signal_short (confluencia de las 3).

    Largo: tendencia alcista (EMA50>EMA100+ADX) + el precio retrocedió a tocar la EMA rápida en
    las últimas `lookback` velas y cierra por encima con vela alcista (rebote).
    Corto: espejo exacto — tendencia bajista, el precio subió a tocar la EMA rápida y cierra por
    debajo con vela bajista (rechazo).
    Ambas comparten la capa de volumen (volumen > su media de 20).
    """
    out = df.copy()

    trend_strength_ok = out["adx"] > cfg.trend.adx_threshold
    out["trend_up_ok"] = (out["ema_fast"] > out["ema_slow"]) & trend_strength_ok
    out["trend_dn_ok"] = (out["ema_fast"] < out["ema_slow"]) & trend_strength_ok

    touched_from_below = _rolling_any(out["low"] <= out["ema_fast"], cfg.pullback.lookback)
    touched_from_above = _rolling_any(out["high"] >= out["ema_fast"], cfg.pullback.lookback)

    reclaimed_up = out["close"] > out["ema_fast"]
    reclaimed_dn = out["close"] < out["ema_fast"]

    if cfg.pullback.require_bullish_candle:
        bullish_candle = out["close"] > out["open"]
        bearish_candle = out["close"] < out["open"]
    else:
        bullish_candle = pd.Series(True, index=out.index)
        bearish_candle = pd.Series(True, index=out.index)

    out["pullback_up_ok"] = touched_from_below & reclaimed_up & bullish_candle
    out["pullback_dn_ok"] = touched_from_above & reclaimed_dn & bearish_candle

    out["volume_ok"] = out["volume"] > out["volume_ma"]

    # Interruptor de régimen: bloquea toda señal (no cuenta como capa) cuando la volatilidad
    # relativa reciente está en el tramo más alto de su propia historia — apunta a
    # correcciones/crashes de alta volatilidad no direccional.
    out["regime_ok"] = out["atr_percentile"] <= cfg.regime.max_volatility_percentile

    layers_long = out[["trend_up_ok", "pullback_up_ok", "volume_ok"]].sum(axis=1)
    layers_short = out[["trend_dn_ok", "pullback_dn_ok", "volume_ok"]].sum(axis=1)

    out["signal_long"] = (layers_long == 3) & out["regime_ok"]
    out["signal_short"] = (layers_short == 3) & out["regime_ok"]

    return out


class ConfluenceStrategy(Strategy):
    """Estrategia largo/corto de confluencia. SL/TP dinámicos vía ATR.

    El sizing usado aquí es riesgo fijo por operación (% de equity / distancia al SL),
    solo para poder correr el backtest. El sizing definitivo (Kelly fraccionado) depende
    del win rate y ratio riesgo/beneficio reales que entregue este mismo backtest.
    """

    risk_per_trade: float = STRATEGY.risk.backtest_risk_per_trade
    sl_atr_mult: float = STRATEGY.risk.sl_atr_mult
    tp_atr_mult: float = STRATEGY.risk.tp_atr_mult
    # "structure": SL = swing_low/swing_high reciente ± sl_atr_mult*ATR (stop estructural).
    # "entry": SL = precio_entrada ± sl_atr_mult*ATR (ratio ATR limpio, ignora estructura).
    # "entry" fue el ancla usada en la calibración final por MAE/MFE (ver FINDINGS.md).
    sl_anchor: str = "entry"

    def init(self) -> None:
        self.signal_long = self.I(lambda: self.data.signal_long, name="signal_long")
        self.signal_short = self.I(lambda: self.data.signal_short, name="signal_short")
        self.atr = self.I(lambda: self.data.atr, name="atr")
        self.swing_low = self.I(lambda: self.data.swing_low, name="swing_low")
        self.swing_high = self.I(lambda: self.data.swing_high, name="swing_high")

    def _size_fraction(self, price: float, risk_per_unit: float) -> float:
        equity_at_risk = self.equity * self.risk_per_trade
        units = equity_at_risk / risk_per_unit
        return min(0.99, (units * price) / self.equity)

    def next(self) -> None:
        if self.position:
            return

        price = self.data.Close[-1]
        atr = self.atr[-1]

        if self.signal_long[-1]:
            anchor = price if self.sl_anchor == "entry" else self.swing_low[-1]
            sl = anchor - self.sl_atr_mult * atr
            tp = price + self.tp_atr_mult * atr
            if sl >= price:
                return  # ancla estructural insuficiente/corrupta (poco histórico al inicio)

            size_fraction = self._size_fraction(price, risk_per_unit=price - sl)
            if size_fraction > 0:
                self.buy(size=size_fraction, sl=sl, tp=tp)

        elif self.signal_short[-1]:
            anchor = price if self.sl_anchor == "entry" else self.swing_high[-1]
            sl = anchor + self.sl_atr_mult * atr
            tp = price - self.tp_atr_mult * atr
            if sl <= price or tp <= 0:
                return

            size_fraction = self._size_fraction(price, risk_per_unit=sl - price)
            if size_fraction > 0:
                self.sell(size=size_fraction, sl=sl, tp=tp)
