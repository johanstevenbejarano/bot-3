"""Descarga y cachea dos fuentes de "flujo institucional", cada una del único exchange donde se
confirmó que hay suficiente historia gratis (ver FINDINGS.md "Línea 5" para el resto de exchanges
probados y por qué se descartaron):

- **Interés abierto** de Bybit (`fetch_open_interest_history`): Binance solo da ~30 días vía API
  pública (ver nota en `funding_data.py`), Bybit sí retiene años. Se usa como proxy del
  posicionamiento agregado del mercado, aplicado sobre el precio de Binance (mismo criterio que
  ya usa `funding_data.py`: una fuente de información distinta, no necesita venir del mismo
  exchange que el precio).
- **Ratio de compra/venta agresiva (taker)** de Binance, extraído directo de las velas crudas
  (`GET /api/v3/klines`, campo 9 = "taker buy base asset volume") -- mismo endpoint y misma
  historia completa que ya usa `data_fetch.py` para OHLCV, así que no hay límite de retención
  nuevo que descubrir acá. Es el sustituto real del "ratio long/short por cuenta" (ese sí tiene
  el mismo límite de ~30 días en todos los exchanges probados) -- mide presión de compra/venta
  real en vez de posición declarada, pero captura la misma idea de fondo.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 2.0
RECORDS_PER_REQUEST = 500  # tope real observado en Bybit para open interest (ver FINDINGS.md)
KLINES_PER_REQUEST = 1000


def _cache_path(symbol: str, kind: str) -> Path:
    safe_symbol = symbol.replace("/", "-")
    return DATA_DIR / f"{safe_symbol}_{kind}.parquet"


def _to_bybit_swap_symbol(symbol: str) -> str:
    """`BTC/USDT` -> `BTC/USDT:USDT` (formato ccxt para perpetuos USDT de Bybit)."""
    return symbol if ":" in symbol else f"{symbol}:USDT"


def _fetch_oi_with_retry(exchange: ccxt.Exchange, symbol: str, since_ms: int) -> list[dict]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return exchange.fetch_open_interest_history(
                symbol, timeframe="1h", since=since_ms, limit=RECORDS_PER_REQUEST
            )
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as exc:
            last_error = exc
            wait = RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "Fetch open interest %s falló (intento %d/%d): %s. Reintentando en %.1fs",
                symbol, attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"No se pudo descargar open interest de {symbol} tras {MAX_RETRIES} intentos") from last_error


def _fetch_oi_range(exchange: ccxt.Exchange, symbol: str, since_ms: int, until_ms: int) -> pd.DataFrame:
    """A diferencia de OHLCV/funding rate, el tamaño real de página de Bybit para este endpoint
    no siempre coincide con `limit` (se observó 200 filas devueltas pidiendo 500) -- una página
    corta NO significa "no hay más historia". Por eso acá se sigue avanzando el cursor mientras
    haya progreso real, en vez de cortar apenas una página viene incompleta.
    """
    all_rows: list[dict] = []
    cursor = since_ms

    while cursor < until_ms:
        rows = _fetch_oi_with_retry(exchange, symbol, cursor)
        rows = [r for r in rows if r["timestamp"] is not None and r["timestamp"] < until_ms]
        if not rows:
            break
        all_rows.extend(rows)
        next_cursor = rows[-1]["timestamp"] + 1
        if next_cursor <= cursor:
            break  # sin avance real -> cortar para no loopear infinito
        cursor = next_cursor
        time.sleep(exchange.rateLimit / 1000)

    if not all_rows:
        return pd.DataFrame(columns=["open_interest"])

    df = pd.DataFrame(
        {
            "timestamp": [r["timestamp"] for r in all_rows],
            "open_interest": [r["openInterestAmount"] for r in all_rows],
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")


def fetch_open_interest_bybit(symbol: str, years: int, force_refresh: bool = False) -> pd.Series:
    """Interés abierto horario de Bybit (perpetuo USDT), cacheado en parquet."""
    cache_file = _cache_path(symbol, "open_interest_bybit")
    exchange = ccxt.bybit({"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "swap"}})
    bybit_symbol = _to_bybit_swap_symbol(symbol)

    since_dt = datetime.now(timezone.utc) - timedelta(days=365 * years)
    since_ms = int(since_dt.timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    cached = pd.DataFrame()
    if cache_file.exists() and not force_refresh:
        cached = pd.read_parquet(cache_file)

    pieces = [cached] if not cached.empty else []

    if cached.empty:
        pieces.append(_fetch_oi_range(exchange, bybit_symbol, since_ms, now_ms))
    else:
        cached_start_ms = int(cached.index[0].timestamp() * 1000)
        cached_end_ms = int(cached.index[-1].timestamp() * 1000) + 1
        if since_ms < cached_start_ms:
            pieces.append(_fetch_oi_range(exchange, bybit_symbol, since_ms, cached_start_ms))
        if cached_end_ms < now_ms:
            pieces.append(_fetch_oi_range(exchange, bybit_symbol, cached_end_ms, now_ms))

    non_empty = [p for p in pieces if not p.empty]
    if not non_empty:
        raise RuntimeError(f"No se obtuvo open interest para {symbol}")

    combined = pd.concat(non_empty)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(cache_file)

    logger.info(
        "%s: %d registros de open interest (Bybit) en caché (%s -> %s)",
        symbol, len(combined), combined.index[0], combined.index[-1],
    )
    return combined["open_interest"]


def _fetch_klines_with_retry(exchange: ccxt.Exchange, symbol: str, since_ms: int) -> list[list]:
    last_error: Exception | None = None
    params = {"symbol": symbol.replace("/", ""), "interval": "1h", "startTime": since_ms, "limit": KLINES_PER_REQUEST}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return exchange.publicGetKlines(params)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as exc:
            last_error = exc
            wait = RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "Fetch klines crudas %s falló (intento %d/%d): %s. Reintentando en %.1fs",
                symbol, attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"No se pudo descargar klines crudas de {symbol} tras {MAX_RETRIES} intentos") from last_error


def _fetch_taker_ratio_range(exchange: ccxt.Exchange, symbol: str, since_ms: int, until_ms: int) -> pd.DataFrame:
    all_rows: list[list] = []
    cursor = since_ms

    while cursor < until_ms:
        rows = _fetch_klines_with_retry(exchange, symbol, cursor)
        rows = [r for r in rows if r[0] < until_ms]
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < KLINES_PER_REQUEST:
            break
        cursor = rows[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_rows:
        return pd.DataFrame(columns=["taker_buy_ratio"])

    timestamps = [r[0] for r in all_rows]
    volume = pd.Series([float(r[5]) for r in all_rows])
    taker_buy_volume = pd.Series([float(r[9]) for r in all_rows])
    # vela sin volumen -> ratio neutro (0.5), no NaN ni división por cero
    ratio = (taker_buy_volume / volume.replace(0, pd.NA)).fillna(0.5)

    df = pd.DataFrame({"timestamp": timestamps, "taker_buy_ratio": ratio.to_numpy(dtype=float)})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")


def fetch_taker_buy_ratio_binance(symbol: str, years: int, force_refresh: bool = False) -> pd.Series:
    """Fracción del volumen de cada vela ejecutada por compradores agresivos (taker buy), de
    Binance spot, cacheada en parquet. 0.5 = compras y ventas agresivas equilibradas."""
    cache_file = _cache_path(symbol, "taker_buy_ratio")
    exchange = ccxt.binance({"enableRateLimit": True})

    since_dt = datetime.now(timezone.utc) - timedelta(days=365 * years)
    since_ms = int(since_dt.timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    cached = pd.DataFrame()
    if cache_file.exists() and not force_refresh:
        cached = pd.read_parquet(cache_file)

    pieces = [cached] if not cached.empty else []

    if cached.empty:
        pieces.append(_fetch_taker_ratio_range(exchange, symbol, since_ms, now_ms))
    else:
        cached_start_ms = int(cached.index[0].timestamp() * 1000)
        cached_end_ms = int(cached.index[-1].timestamp() * 1000) + 1
        if since_ms < cached_start_ms:
            pieces.append(_fetch_taker_ratio_range(exchange, symbol, since_ms, cached_start_ms))
        if cached_end_ms < now_ms:
            pieces.append(_fetch_taker_ratio_range(exchange, symbol, cached_end_ms, now_ms))

    non_empty = [p for p in pieces if not p.empty]
    if not non_empty:
        raise RuntimeError(f"No se obtuvo taker buy ratio para {symbol}")

    combined = pd.concat(non_empty)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(cache_file)

    logger.info(
        "%s: %d registros de taker buy ratio (Binance) en caché (%s -> %s)",
        symbol, len(combined), combined.index[0], combined.index[-1],
    )
    return combined["taker_buy_ratio"]
