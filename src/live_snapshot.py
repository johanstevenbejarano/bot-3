"""Trae datos frescos de Binance (sin caché en disco — cada corrida es efímera, pensado para
correr en un agente en la nube) y arma el HTML completo del dashboard de confluencia.

A diferencia de `data_fetch.py` (pensado para backtesting, con caché en parquet), acá no hace
falta guardar nada entre corridas ni acumular años de historia — solo los últimos ~120 días
alcanzan para todos los indicadores en vivo (el lookback más largo es el percentil de ATR, 90
días). Esto evita la dependencia de pyarrow y evita reescribir un caché que nadie va a releer.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd

from src.breakout_indicators import add_breakout_indicators
from src.breakout_strategy import compute_layers as breakout_layers
from src.config import BREAKOUT_STRATEGY, FUNDING_STRATEGY, MEANREV_STRATEGY, STRATEGY
from src.dashboard_render import compute_risk_levels, format_price, make_sparkline_svg, pct_change
from src.funding_indicators import align_funding_to_1h, compute_funding_percentile
from src.indicators import add_indicators
from src.meanrev_indicators import add_meanrev_indicators
from src.meanrev_strategy import compute_layers as meanrev_layers
from src.strategy import compute_layers as trend_layers

SYMBOLS = ("BTC/USDT", "ETH/USDT")
LOOKBACK_DAYS = 120
SPARKLINE_HOURS = 72


def fetch_recent_ohlcv(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Descarga directa (sin caché) de las últimas `days` de velas de 1h."""
    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    rows: list[list[float]] = []
    cursor = since_ms
    while cursor < now_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe="1h", since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        cursor = batch[-1][0] + 1

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")


def fetch_recent_funding(symbol: str, days: int = LOOKBACK_DAYS) -> pd.Series:
    exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    rows: list[dict] = []
    cursor = since_ms
    while cursor < now_ms:
        batch = exchange.fetch_funding_rate_history(symbol, since=cursor, limit=1000)
        batch = [r for r in batch if r["timestamp"] is not None]
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        cursor = batch[-1]["timestamp"] + 1

    df = pd.DataFrame({"timestamp": [r["timestamp"] for r in rows], "funding_rate": [r["fundingRate"] for r in rows]})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")["funding_rate"]


def compute_symbol_snapshot(symbol: str, raw: pd.DataFrame, funding_rate: pd.Series) -> dict:
    last = raw.iloc[-1]
    price = float(last["close"])

    trend_df = trend_layers(add_indicators(raw, STRATEGY), STRATEGY)
    t = trend_df.iloc[-1]
    mr_df = meanrev_layers(add_meanrev_indicators(raw, MEANREV_STRATEGY), MEANREV_STRATEGY)
    m = mr_df.iloc[-1]
    bo_df = breakout_layers(add_breakout_indicators(raw, BREAKOUT_STRATEGY), BREAKOUT_STRATEGY)
    b = bo_df.iloc[-1]

    funding_pct_series = compute_funding_percentile(funding_rate, FUNDING_STRATEGY.lookback_periods)
    aligned = align_funding_to_1h(raw, funding_pct_series)
    funding_percentile = float(aligned["funding_percentile"].iloc[-1]) if not aligned["funding_percentile"].isna().all() else float("nan")

    atr = float(t["atr"])
    closes_72h = raw["close"].tail(SPARKLINE_HOURS).round(2).tolist()
    spark = make_sparkline_svg(closes_72h)
    change_72h = pct_change(closes_72h)
    risk = compute_risk_levels(price, atr, BREAKOUT_STRATEGY.risk.sl_atr_mult, BREAKOUT_STRATEGY.risk.tp_atr_mult)

    return {
        "symbol": symbol,
        "price": price,
        "price_fmt": format_price(price),
        "change_72h": change_72h,
        "adx": float(t["adx"]),
        "atr_pct_of_price": atr / price * 100,
        "funding_percentile": funding_percentile,
        "sparkline": spark,
        "risk": risk,
        "trend_flags": [bool(t["trend_up_ok"]) or bool(t["trend_dn_ok"]), bool(t["pullback_up_ok"]) or bool(t["pullback_dn_ok"]), bool(t["volume_ok"])],
        "meanrev_flags": [bool(m["touched_lower"]) or bool(m["touched_upper"]), bool(m["rsi_oversold"]) or bool(m["rsi_overbought"]), bool(m["volume_ok"])],
        "breakout_flags": [bool(b["breakout_up"]) or bool(b["breakout_down"]), bool(b["volume_ok"])],
        "signal_long": bool(t["signal_long"]) or bool(m["signal_long"]) or bool(b["signal_long"]),
        "signal_short": bool(t["signal_short"]) or bool(m["signal_short"]) or bool(b["signal_short"]),
        "active_names": [
            name for name, cond in [
                ("tendencia", bool(t["signal_long"]) or bool(t["signal_short"])),
                ("reversión", bool(m["signal_long"]) or bool(m["signal_short"])),
                ("breakout", bool(b["signal_long"]) or bool(b["signal_short"])),
            ] if cond
        ],
    }


def build_snapshot() -> dict:
    raw_by_symbol = {s: fetch_recent_ohlcv(s) for s in SYMBOLS}
    funding_by_symbol = {s: fetch_recent_funding(s) for s in SYMBOLS}

    symbols_data = {
        s: compute_symbol_snapshot(s, raw_by_symbol[s], funding_by_symbol[s]) for s in SYMBOLS
    }

    btc_ret = raw_by_symbol["BTC/USDT"]["close"].pct_change().tail(720)
    eth_ret = raw_by_symbol["ETH/USDT"]["close"].pct_change().tail(720)
    common = btc_ret.index.intersection(eth_ret.index)
    correlation = float(btc_ret.loc[common].corr(eth_ret.loc[common]))

    return {
        "generated_at": datetime.now(timezone.utc),
        "symbols": symbols_data,
        "correlation": correlation,
    }


def main(output_path: str = "dashboard_output.html") -> None:
    from src.dashboard_template import render_html

    snap = build_snapshot()
    html = render_html(snap)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    for s, d in snap["symbols"].items():
        print(f"{s}: {d['price_fmt']} | signal_long={d['signal_long']} signal_short={d['signal_short']}")
    print(f"HTML escrito en {output_path}")


if __name__ == "__main__":
    main()
