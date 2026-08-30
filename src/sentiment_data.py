"""Descarga y cachea el índice Fear & Greed (alternative.me) -- sentimiento agregado del mercado
cripto completo (no específico de BTC o ETH), gratis y sin autenticación, con historia diaria
desde el 1 de febrero de 2018. A diferencia de open interest/ratio long-short (ver FINDINGS.md
"Línea 5"), este endpoint no tiene límite de retención: `limit=0` devuelve todo de una sola vez.
"""
from __future__ import annotations

import logging

import pandas as pd
import requests

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

API_URL = "https://api.alternative.me/fng/"
_CACHE_FILE = DATA_DIR / "fear_greed_index.parquet"


def fetch_fear_greed_index(force_refresh: bool = False) -> pd.Series:
    """Serie diaria (índice UTC a medianoche) del valor 0-100 del Fear & Greed Index, cacheada."""
    if _CACHE_FILE.exists() and not force_refresh:
        cached = pd.read_parquet(_CACHE_FILE)
        return cached["value"]

    response = requests.get(API_URL, params={"limit": 0, "format": "json"}, timeout=20)
    response.raise_for_status()
    rows = response.json()["data"]

    df = pd.DataFrame(
        {
            "timestamp": [int(r["timestamp"]) for r in rows],
            "value": [float(r["value"]) for r in rows],
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    df.to_parquet(_CACHE_FILE)
    logger.info("Fear & Greed Index: %d registros en caché (%s -> %s)", len(df), df.index[0], df.index[-1])
    return df["value"]
