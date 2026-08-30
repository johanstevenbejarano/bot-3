"""Descarga y cachea OHLCV del perpetuo (Binance Futures) -- mismo patrón que `data_fetch.py`
(spot), necesario para calcular la base (spot vs. perpetuo) del arbitraje de funding rate.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd

from src.config import BACKTEST, DATA_DIR

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 2.0
CANDLES_PER_REQUEST = 1000


def _cache_path(symbol: str) -> Path:
    safe_symbol = symbol.replace("/", "-")
    return DATA_DIR / f"{safe_symbol}_perp_1h.parquet"


def _make_exchange() -> ccxt.binance:
    return ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})


def _fetch_with_retry(exchange: ccxt.Exchange, symbol: str, since_ms: int) -> list[list[float]]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe="1h", since=since_ms, limit=CANDLES_PER_REQUEST)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as exc:
            last_error = exc
            wait = RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "Fetch perp %s falló (intento %d/%d): %s. Reintentando en %.1fs",
                symbol, attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"No se pudo descargar perpetuo de {symbol} tras {MAX_RETRIES} intentos") from last_error


def _fetch_range(exchange: ccxt.Exchange, symbol: str, since_ms: int, until_ms: int) -> pd.DataFrame:
    all_rows: list[list[float]] = []
    cursor = since_ms

    while cursor < until_ms:
        rows = _fetch_with_retry(exchange, symbol, cursor)
        if not rows:
            break
        rows = [r for r in rows if r[0] < until_ms]
        all_rows.extend(rows)
        if not rows or len(rows) < CANDLES_PER_REQUEST:
            break
        cursor = rows[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_rows:
        return pd.DataFrame(columns=["close"])

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")


def fetch_perp_close(symbol: str, years: int, force_refresh: bool = False) -> pd.Series:
    """Cierre horario del contrato perpetuo, cacheado en parquet."""
    cache_file = _cache_path(symbol)
    exchange = _make_exchange()

    since_dt = datetime.now(timezone.utc) - timedelta(days=365 * years)
    since_ms = int(since_dt.timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    cached = pd.DataFrame()
    if cache_file.exists() and not force_refresh:
        cached = pd.read_parquet(cache_file)

    pieces = [cached] if not cached.empty else []

    if cached.empty:
        pieces.append(_fetch_range(exchange, symbol, since_ms, now_ms))
    else:
        cached_start_ms = int(cached.index[0].timestamp() * 1000)
        cached_end_ms = int(cached.index[-1].timestamp() * 1000) + 1
        if since_ms < cached_start_ms:
            pieces.append(_fetch_range(exchange, symbol, since_ms, cached_start_ms))
        if cached_end_ms < now_ms:
            pieces.append(_fetch_range(exchange, symbol, cached_end_ms, now_ms))

    non_empty = [p for p in pieces if not p.empty]
    if not non_empty:
        raise RuntimeError(f"No se obtuvo perpetuo para {symbol}")

    combined = pd.concat(non_empty)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(cache_file)

    logger.info(
        "%s: %d velas de perpetuo en caché (%s -> %s)",
        symbol, len(combined), combined.index[0], combined.index[-1],
    )
    return combined["close"]
