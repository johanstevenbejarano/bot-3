"""Descarga y cachea el índice del dólar (DXY) vía la API pública de gráficos de Yahoo Finance --
sin autenticación, historia diaria completa desde antes del inicio de los datos de Binance. Es la
correlación macro más citada para cripto ("dólar fuerte, activos de riesgo débiles"), una fuente
de información genuinamente distinta de todo lo probado hasta ahora (precio/volumen/derivados/
sentimiento cripto): mercados tradicionales.
"""
from __future__ import annotations

import logging

import pandas as pd
import requests

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
DXY_TICKER = "DX-Y.NYB"
_CACHE_FILE = DATA_DIR / "dxy_daily.parquet"


def fetch_dxy_daily(years: int = 9, force_refresh: bool = False) -> pd.Series:
    """Cierre diario del índice del dólar (DXY), cacheado en parquet."""
    if _CACHE_FILE.exists() and not force_refresh:
        cached = pd.read_parquet(_CACHE_FILE)
        return cached["close"]

    now = pd.Timestamp.now(tz="UTC")
    period1 = int((now - pd.Timedelta(days=365 * years)).timestamp())
    period2 = int(now.timestamp())

    response = requests.get(
        YAHOO_CHART_URL.format(ticker=DXY_TICKER),
        params={"period1": period1, "period2": period2, "interval": "1d"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()
    result = response.json()["chart"]["result"][0]

    timestamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]

    df = pd.DataFrame({"timestamp": timestamps, "close": closes})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df.dropna(subset=["close"])  # feriados/fines de semana bursátiles -> hueco, no vela

    df.to_parquet(_CACHE_FILE)
    logger.info("DXY: %d registros diarios en caché (%s -> %s)", len(df), df.index[0], df.index[-1])
    return df["close"]
