"""Trae datos frescos de Binance (sin caché en disco — cada corrida es efímera, pensado para
correr en un agente en la nube) y arma el HTML completo del dashboard de confluencia.

A diferencia de `data_fetch.py` (pensado para backtesting, con caché en parquet), acá no hace
falta guardar nada entre corridas ni acumular años de historia — solo los últimos ~120 días
alcanzan para todos los indicadores en vivo (el lookback más largo es el percentil de ATR, 90
días). Esto evita la dependencia de pyarrow y evita reescribir un caché que nadie va a releer.

Sobre `_patch_requests_ca_bundle_for_sandbox`: el agente en la nube intercepta el tráfico HTTPS
con un proxy TLS propio y expone su certificado vía `REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE`. ccxt,
sin embargo, siempre le pasa a `requests` un `verify=True` (nunca un path propio) en cada
request (ver `Exchange.fetch`) -- por eso `trust_env` de requests no sirve acá: `requests` solo
consulta esas variables de entorno cuando `verify` no es exactamente `True`. La única forma de
que la verificación use el certificado del proxy es reapuntar directamente el bundle por
defecto de `requests` (`requests.adapters.DEFAULT_CA_BUNDLE_PATH`, ver `cert_verify` en esa
librería). Sin efecto fuera de ese sandbox (si esas variables no existen, no cambia nada).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd
import requests.adapters

from src.breakout_strategy import compute_layers as breakout_layers
from src.config import BREAKOUT_STRATEGY, FUNDING_STRATEGY, MEANREV_STRATEGY, STRATEGY
from src.dashboard_render import compute_risk_levels, format_price, make_sparkline_svg, pct_change
from src.funding_indicators import align_funding_to_1h, compute_funding_percentile
from src.meanrev_strategy import compute_layers as meanrev_layers
from src.strategy import compute_layers as trend_layers
from src.ta_free_indicators import adx as ta_free_adx
from src.ta_free_indicators import atr as ta_free_atr
from src.ta_free_indicators import bollinger_bands, ema, rsi

SYMBOLS = ("BTC/USDT", "ETH/USDT")
LOOKBACK_DAYS = 120
SPARKLINE_HOURS = 72


def _patch_requests_ca_bundle_for_sandbox() -> None:
    """Ver docstring del módulo. No-op si el sandbox no expone su propio CA bundle."""
    ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    if ca_bundle and os.path.isfile(ca_bundle):
        requests.adapters.DEFAULT_CA_BUNDLE_PATH = ca_bundle


_patch_requests_ca_bundle_for_sandbox()


def _add_trend_indicators(df: pd.DataFrame, cfg=STRATEGY) -> pd.DataFrame:
    """Equivalente sin `ta` de `src.indicators.add_indicators` (mismas columnas/fórmulas)."""
    out = df.copy()
    out["ema_fast"] = ema(out["close"], cfg.trend.ema_fast)
    out["ema_slow"] = ema(out["close"], cfg.trend.ema_slow)
    out["adx"] = ta_free_adx(out["high"], out["low"], out["close"], cfg.trend.adx_period)
    out["atr"] = ta_free_atr(out["high"], out["low"], out["close"], cfg.risk.atr_period)
    out["volume_ma"] = out["volume"].rolling(window=cfg.volume.volume_ma_period).mean()
    out["swing_low"] = out["low"].rolling(window=cfg.risk.swing_lookback).min()
    out["swing_high"] = out["high"].rolling(window=cfg.risk.swing_lookback).max()

    atr_pct = out["atr"] / out["close"]
    out["atr_percentile"] = atr_pct.rolling(
        window=cfg.regime.lookback_bars, min_periods=cfg.regime.lookback_bars // 2
    ).rank(pct=True)

    return out.dropna()


def _add_meanrev_indicators(df: pd.DataFrame, cfg=MEANREV_STRATEGY) -> pd.DataFrame:
    """Equivalente sin `ta` de `src.meanrev_indicators.add_meanrev_indicators`."""
    out = df.copy()
    out["bb_basis"], out["bb_upper"], out["bb_lower"] = bollinger_bands(
        out["close"], cfg.bollinger.period, cfg.bollinger.num_std
    )
    out["rsi"] = rsi(out["close"], cfg.rsi.period)
    out["atr"] = ta_free_atr(out["high"], out["low"], out["close"], cfg.risk.atr_period)
    out["volume_ma"] = out["volume"].rolling(window=cfg.volume.volume_ma_period).mean()
    return out.dropna()


def _add_breakout_indicators(df: pd.DataFrame, cfg=BREAKOUT_STRATEGY) -> pd.DataFrame:
    """Equivalente sin `ta` de `src.breakout_indicators.add_breakout_indicators`."""
    out = df.copy()
    out["donchian_high"] = out["high"].rolling(window=cfg.donchian.period).max().shift(1)
    out["donchian_low"] = out["low"].rolling(window=cfg.donchian.period).min().shift(1)
    out["atr"] = ta_free_atr(out["high"], out["low"], out["close"], cfg.risk.atr_period)
    out["volume_ma"] = out["volume"].rolling(window=cfg.volume.volume_ma_period).mean()
    return out.dropna()


def fetch_recent_ohlcv(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Descarga directa (sin caché) de las últimas `days` de velas de 1h."""
    # ccxt trae por defecto exchangeInfo de spot + futuros USD-M + futuros COIN-M en
    # load_markets(), aunque acá solo hace falta spot -- "fetchMarkets": ["spot"] evita que
    # también golpee fapi.binance.com/dapi.binance.com (bloqueados igual que api.binance.com,
    # ver más abajo) solo para armar una lista de mercados que ni se usa.
    exchange = ccxt.binance(
        {"enableRateLimit": True, "timeout": 30000, "options": {"fetchMarkets": ["spot"]}}
    )
    # api.binance.com devuelve 451 (bloqueo por ubicación/tipo de IP -- ver "Eligibility" en sus
    # términos) desde IPs de centros de datos en la nube. data-api.binance.vision es el espejo
    # público de solo-lectura que Binance ofrece justamente para esto (sin ese bloqueo). Solo
    # cubre datos de mercado (klines/exchangeInfo), no cuentas ni trading.
    exchange.urls["api"]["public"] = "https://data-api.binance.vision/api/v3"
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
    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
            "timeout": 30000,
            "options": {"defaultType": "future", "fetchMarkets": ["linear"]},
        }
    )
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

    trend_df = trend_layers(_add_trend_indicators(raw, STRATEGY), STRATEGY)
    t = trend_df.iloc[-1]
    mr_df = meanrev_layers(_add_meanrev_indicators(raw, MEANREV_STRATEGY), MEANREV_STRATEGY)
    m = mr_df.iloc[-1]
    bo_df = breakout_layers(_add_breakout_indicators(raw, BREAKOUT_STRATEGY), BREAKOUT_STRATEGY)
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


def _fetch_funding_best_effort(symbol: str) -> pd.Series:
    """El funding rate es solo un dato de contexto (la estrategia basada en él no mostró ventaja
    validada, ver FINDINGS.md) -- si el endpoint de futuros falla (ej. geo-bloqueo de Binance en
    algunos entornos), el dashboard debe seguir funcionando sin ese dato puntual, no romperse
    entero. `compute_funding_percentile`/`align_funding_to_1h` ya manejan una serie vacía sin
    error (terminan en NaN, que la plantilla muestra como "s/d").
    """
    try:
        return fetch_recent_funding(symbol)
    except Exception:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))


def build_snapshot() -> dict:
    raw_by_symbol = {s: fetch_recent_ohlcv(s) for s in SYMBOLS}
    funding_by_symbol = {s: _fetch_funding_best_effort(s) for s in SYMBOLS}

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

    print("\n=== Datos para la sección 'Interpretación' (ver instrucciones de la rutina) ===")
    for s, d in snap["symbols"].items():
        funding_txt = (
            "sin dato"
            if d["funding_percentile"] != d["funding_percentile"]
            else f"percentil {d['funding_percentile'] * 100:.0f} de su propia historia reciente"
        )
        senal = "LARGO" if d["signal_long"] else "CORTO" if d["signal_short"] else "ninguna"
        print(f"\n{s}:")
        print(f"  precio: {d['price_fmt']} | cambio 72h: {d['change_72h']:+.2f}%")
        print(f"  ADX: {d['adx']:.1f} (por encima de 25 se considera tendencia fuerte)")
        print(f"  ATR: {d['atr_pct_of_price']:.2f}% del precio (volatilidad relativa)")
        print(f"  funding rate: {funding_txt}")
        print(f"  capas tendencia [tendencia_ok, retroceso_ok, volumen_ok]: {d['trend_flags']}")
        print(f"  capas reversión [toque_banda_ok, rsi_extremo_ok, volumen_ok]: {d['meanrev_flags']}")
        print(f"  capas breakout [ruptura_canal_ok, volumen_ok]: {d['breakout_flags']}")
        print(f"  señal de confluencia activa ahora: {senal} ({', '.join(d['active_names']) or 'ninguna estrategia'})")


if __name__ == "__main__":
    main()
